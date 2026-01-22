#!/usr/bin/env python
# -*- coding: utf-8 -*-
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Query
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from qdrant_client import QdrantClient, models
import httpx
from fastembed import SparseTextEmbedding
import numpy as np
import json
import openpyxl
import io
import logging
import re
import asyncio
import os
import qdrant_client as qc
import fastembed
import fastapi
import importlib.metadata

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

logger.info("Starting API...")
try:
    logger.info(f"Qdrant Client version: {importlib.metadata.version('qdrant-client')}")
except Exception:
    logger.info("Qdrant Client version: unknown")
try:
    logger.info(f"FastEmbed version: {importlib.metadata.version('fastembed')}")
except Exception:
    logger.info("FastEmbed version: unknown")
try:
    logger.info(f"FastAPI version: {importlib.metadata.version('fastapi')}")
except Exception:
    logger.info("FastAPI version: unknown")

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
async def read_root():
    with open("static/index.html") as f:
        return f.read()

qdrant_client = QdrantClient(host=os.getenv("QDRANT_HOST", "qdrant"), port=int(os.getenv("QDRANT_PORT", "6333")))
ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")
ollama_api_url = f"{ollama_base_url}/api/embeddings"
ollama_batch_api_url = f"{ollama_base_url}/api/embed"

# Initialize FastEmbed Sparse Model
# Using a standard SPLADE model which acts as a learned BM25 replacement.
sparse_embedding_model = SparseTextEmbedding(model_name="prithivida/Splade_PP_en_v1")

def create_collection(collection_name: str):
    try:
        qdrant_client.get_collection(collection_name=collection_name)
    except Exception:
        qdrant_client.recreate_collection(
            collection_name=collection_name,
            vectors_config={
                "text-dense": models.VectorParams(size=1024, distance=models.Distance.COSINE),
            },
            sparse_vectors_config={
                "text-sparse": models.SparseVectorParams(
                    index=models.SparseIndexParams(
                        on_disk=False,
                    )
                )
            },
        )

async def get_ollama_embeddings(texts, concurrency=4):
    async with httpx.AsyncClient(timeout=300.0) as client:
        # Try batch endpoint first (if supported by Ollama)
        try:
            batch_response = await client.post(
                ollama_batch_api_url,
                json={"model": "bge-m3", "input": texts},
            )
            if batch_response.status_code < 400:
                data = batch_response.json()
                if "embeddings" in data:
                    return data["embeddings"]
        except httpx.HTTPError:
            pass

        # Fallback: parallel single requests to /api/embeddings
        semaphore = asyncio.Semaphore(concurrency)

        async def fetch_one(text):
            async with semaphore:
                response = await client.post(
                    ollama_api_url,
                    json={"model": "bge-m3", "prompt": text},
                )
                response.raise_for_status()
                return response.json()["embedding"]

        return await asyncio.gather(*(fetch_one(text) for text in texts))

async def get_ollama_embedding(text: str):
    embeddings = await get_ollama_embeddings([text])
    return embeddings[0]

def get_sparse_embedding(text: str):
    # FastEmbed returns a generator of sparse embeddings
    embeddings = list(sparse_embedding_model.embed([text]))
    return embeddings[0]

@app.post("/upload_processed_xlsx")
async def upload_processed_xlsx(
    file: UploadFile = File(...),
    skip_rows: int = Form(...),
    mappings: str = Form(...),
    collection_name: str = Form(...),
    batch_size: int = Form(16),
):
    logger.info(f"Received request to /upload_processed_xlsx for collection: {collection_name}")
    logger.info(f"skip_rows: {skip_rows}, mappings: {mappings}, batch_size: {batch_size}")
    create_collection(collection_name)

    try:
        mappings = json.loads(mappings)
        if not mappings:
            raise HTTPException(status_code=400, detail="Mappings cannot be empty")
        logger.info(f"Parsed mappings: {mappings}")
        contents = await file.read()
        workbook = openpyxl.load_workbook(io.BytesIO(contents))
        sheet = workbook.active

        rows = list(sheet.iter_rows(min_row=skip_rows + 2, values_only=True))
        documents = []
        for row in rows:
            # Safely handle None values in row
            def get_val(idx):
                if idx < len(row) and row[idx] is not None:
                    return str(row[idx])
                return ""

            text_to_embed = (
                f"Артикул - {get_val(mappings['Артикул'])}, "
                f"Наименование - {get_val(mappings['Наименование'])}, "
                f"Тариф с НДС, руб - {get_val(mappings['Тариф с НДС, руб'])}, "
                f"Имя файла - {file.filename}"
            )
            documents.append(text_to_embed)

        points = []

        # Batch generation for sparse embeddings (FastEmbed is already efficient)
        sparse_vectors = list(sparse_embedding_model.embed(documents))

        # Batch dense embeddings via Ollama
        for start in range(0, len(documents), batch_size):
            batch_docs = documents[start : start + batch_size]
            dense_vectors = await get_ollama_embeddings(batch_docs)

            for offset, dense_vector in enumerate(dense_vectors):
                i = start + offset
                sparse_vector = sparse_vectors[i]

                qdrant_sparse_vector = models.SparseVector(
                    indices=sparse_vector.indices.tolist(),
                    values=sparse_vector.values.tolist(),
                )

                row_data = rows[i]
                payload = {}
                for header, col_idx in mappings.items():
                    if col_idx < len(row_data):
                        payload[header] = row_data[col_idx]
                payload["Остаток"] = ""
                payload["Имя файла"] = file.filename

                points.append(
                    models.PointStruct(
                        id=i,
                        vector={
                            "text-dense": dense_vector,
                            "text-sparse": qdrant_sparse_vector,
                        },
                        payload=payload,
                    )
                )

        qdrant_client.upsert(
            collection_name=collection_name,
            wait=True,
            points=points,
        )
        return {"status": "success", "indexed_rows": len(documents)}
    except Exception as e:
        logger.error(f"Error processing file: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/search")
async def search(query: str = Query(...), collection_name: str = Query(...)):
    # Dense vector for semantic search from Ollama
    dense_vector = await get_ollama_embedding(query)

    # Sparse vector from FastEmbed (No more re-indexing!)
    sparse_vector_gen = list(sparse_embedding_model.embed([query]))[0]
    sparse_vector = models.SparseVector(
        indices=sparse_vector_gen.indices.tolist(), 
        values=sparse_vector_gen.values.tolist()
    )
    
    # Hybrid search using Query API (Prefetch + Fusion RRF) - Requires qdrant-client >= 1.10.0
    try:
        search_result = qdrant_client.query_points(
            collection_name=collection_name,
            prefetch=[
                models.Prefetch(
                    query=sparse_vector,
                    using="text-sparse",
                    limit=20,
                ),
                models.Prefetch(
                    query=dense_vector,
                    using="text-dense",
                    limit=20,
                ),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=10,
            with_payload=True,
        )
        return {"results": search_result.points}
    except Exception as e:
        logger.error(f"Search failed: {e}", exc_info=True)
        # Return error details to the client for easier debugging
        raise HTTPException(status_code=500, detail=str(e))
