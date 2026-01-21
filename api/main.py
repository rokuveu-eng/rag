#!/usr/bin/env python
# -*- coding: utf-8 -*-
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Query
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from qdrant_client import QdrantClient, models
import httpx
from fastembed import TextEmbedding
import json
import openpyxl
import io
import logging
import os
import numpy as np
from typing import List

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Serve static files
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
async def read_root():
    with open("static/index.html") as f:
        return f.read()

# Initialize clients and models
qdrant_client = QdrantClient(host="qdrant", port=6333)
# Use BAAI/bge-base-en-v1.5 for dense and PRF for sparse
embedding_model = TextEmbedding(model_name="intfloat/multilingual-e5-large", max_length=512)

def create_collection(collection_name: str):
    try:
        qdrant_client.get_collection(collection_name=collection_name)
        logger.info(f"Collection '{collection_name}' already exists.")
    except Exception:
        logger.info(f"Collection '{collection_name}' not found. Creating new collection.")
        qdrant_client.recreate_collection(
            collection_name=collection_name,
            vectors_config={
                "dense": models.VectorParams(size=1024, distance=models.Distance.COSINE),
            },
            sparse_vectors_config={
                "sparse": models.SparseVectorParams(
                    index=models.SparseIndexParams(
                        on_disk=False,
                    )
                )
            },
        )
        logger.info(f"Collection '{collection_name}' created.")

@app.post("/upload_processed_xlsx")
async def upload_processed_xlsx(file: UploadFile = File(...), skip_rows: int = Form(...), mappings: str = Form(...), collection_name: str = Form(...)):
    logger.info(f"Received request to /upload_processed_xlsx for collection: {collection_name}")
    create_collection(collection_name)
    
    try:
        mappings = json.loads(mappings)
        contents = await file.read()
        workbook = openpyxl.load_workbook(io.BytesIO(contents))
        sheet = workbook.active

        documents = []
        payloads = []
        rows = list(sheet.iter_rows(min_row=skip_rows + 2, values_only=True))
        for row in rows:
            text_to_embed = f"Артикул - {row[mappings['Артикул']]}, Наименование - {row[mappings['Наименование']]}, Тариф с НДС, руб - {row[mappings['Тариф с НДС, руб']]}, Имя файла - {file.filename}"
            documents.append(text_to_embed)
            payload = {}
            for header, col_idx in mappings.items():
                payload[header] = row[col_idx]
            payload["Остаток"] = ""
            payloads.append(payload)

        logger.info(f"Generating embeddings for {len(documents)} documents...")
        # Generate both dense and sparse embeddings
        embeddings = list(embedding_model.embed(documents, batch_size=32))
        dense_embeddings = [e.tolist() for e in embeddings]
        sparse_embeddings = list(embedding_model.embed(documents, batch_size=32, sparse=True))

        points = []
        for i, (dense_emb, sparse_emb, payload) in enumerate(zip(dense_embeddings, sparse_embeddings, payloads)):
            points.append(models.PointStruct(
                id=i,  # Simple sequential IDs
                vector={
                    "dense": dense_emb,
                    "sparse": models.SparseVector(indices=sparse_emb.indices.tolist(), values=sparse_emb.values.tolist()),
                },
                payload=payload
            ))

        logger.info(f"Upserting {len(points)} points to Qdrant...")
        qdrant_client.upsert(
            collection_name=collection_name,
            wait=True,
            points=points,
        )
        logger.info("Upsert complete.")
        return {"status": "success", "indexed_rows": len(documents)}
    except Exception as e:
        logger.error(f"Error processing file: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/search")
async def search(query: str = Query(...), collection_name: str = Query(...)):
    logger.info(f"Received search query: '{query}' for collection: '{collection_name}'")

    # Generate dense and sparse vectors for the query
    query_embeddings = list(embedding_model.embed([query], sparse=False))
    query_dense_vector = query_embeddings[0].tolist()
    
    query_sparse_embeddings = list(embedding_model.embed([query], sparse=True))
    query_sparse_vector = models.SparseVector(
        indices=query_sparse_embeddings[0].indices.tolist(), 
        values=query_sparse_embeddings[0].values.tolist()
    )

    # Hybrid search request
    dense_request = models.SearchRequest(
        vector=models.NamedVector(
            name="dense",
            vector=query_dense_vector,
        ),
        limit=5,
        with_payload=True,
    )
    sparse_request = models.SearchRequest(
        vector=models.NamedSparseVector(
            name="sparse",
            vector=query_sparse_vector,
        ),
        limit=5,
        with_payload=True,
    )

    logger.info("Performing hybrid search in Qdrant...")
    search_result = qdrant_client.search_batch(
        collection_name=collection_name,
        requests=[dense_request, sparse_request],
    )
    logger.info("Search complete.")

    return {
        "dense_search_results": search_result[0],
        "sparse_search_results": search_result[1]
    }
