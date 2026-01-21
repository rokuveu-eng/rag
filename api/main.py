#!/usr/bin/env python
# -*- coding: utf-8 -*-
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Query
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from qdrant_client import QdrantClient, models
from qdrant_client.http.models import UpdateStatus
import httpx
from sparse_dot_bm25 import bm25
import numpy as np
import json
import openpyxl
import io
import logging
import re

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
async def read_root():
    with open("static/index.html") as f:
        return f.read()

qdrant_client = QdrantClient(host="qdrant", port=6333)
ollama_api_url = "http://ollama:11434/api/embeddings"

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

async def get_embedding(text: str):
    async with httpx.AsyncClient(timeout=300.0) as client:
        response = await client.post(ollama_api_url, json={"model": "bge-m3", "prompt": text})
        response.raise_for_status()
        return response.json()["embedding"]

def preprocess_text(text: str) -> list[str]:
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    return text.split()


@app.post("/upload_processed_xlsx")
async def upload_processed_xlsx(file: UploadFile = File(...), skip_rows: int = Form(...), mappings: str = Form(...), collection_name: str = Form(...)):
    logger.info(f"Received request to /upload_processed_xlsx for collection: {collection_name}")
    logger.info(f"skip_rows: {skip_rows}, mappings: {mappings}")
    create_collection(collection_name)
    
    try:
        mappings = json.loads(mappings)
        if not mappings:
            raise HTTPException(status_code=400, detail="Mappings cannot be empty")
        logger.info(f"Parsed mappings: {mappings}")
        contents = await file.read()
        workbook = openpyxl.load_workbook(io.BytesIO(contents))
        sheet = workbook.active

        documents = []
        for row in sheet.iter_rows(min_row=skip_rows + 2, values_only=True):
            text_to_embed = f"Артикул - {row[mappings['Артикул']]}, Наименование - {row[mappings['Наименование']]}, Тариф с НДС, руб - {row[mappings['Тариф с НДС, руб']]}, Имя файла - {file.filename}"
            documents.append(text_to_embed)
        
        # Create BM25 model
        processed_docs = [preprocess_text(doc) for doc in documents]
        bm25_model = bm25.BM25(processed_docs)
        
        points = []
        for i, (doc, processed_doc) in enumerate(zip(documents, processed_docs)):
            # Dense vector
            dense_vector = await get_embedding(doc)
            
            # Sparse vector
            sparse_vector_indices, sparse_vector_values = bm25_model.get_doc_vector(processed_doc)
            sparse_vector = models.SparseVector(indices=sparse_vector_indices.tolist(), values=sparse_vector_values.tolist())
            
            # Payload
            row = list(sheet.iter_rows(min_row=skip_rows + 2, values_only=True))[i]
            payload = {}
            for header, col_idx in mappings.items():
                payload[header] = row[col_idx]
            payload["Остаток"] = ""
            
            points.append(models.PointStruct(
                id=i,
                vector={
                    "text-dense": dense_vector,
                    "text-sparse": sparse_vector
                },
                payload=payload
            ))

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
    # Dense vector for semantic search
    dense_vector = await get_embedding(query)

    # Sparse vector for keyword search
    processed_query = preprocess_text(query)
    
    # In a real application, you would load the bm25 model from a saved state.
    # For simplicity, we are re-creating it here from all docs in the collection.
    all_docs = qdrant_client.scroll(collection_name=collection_name, limit=10000, with_payload=True) # Adjust limit as needed
    corpus = [point.payload["Наименование"] for point in all_docs[0]] # Or whatever field you used for BM25
    processed_corpus = [preprocess_text(doc) for doc in corpus]
    bm25_model = bm25.BM25(processed_corpus)
    
    sparse_vector_indices, sparse_vector_values = bm25_model.get_doc_vector(processed_query)
    sparse_vector = models.SparseVector(indices=sparse_vector_indices.tolist(), values=sparse_vector_values.tolist())
    
    # Hybrid search
    search_result = qdrant_client.search_batch(
        collection_name=collection_name,
        requests=[
            models.SearchRequest(
                vector=models.NamedVector(
                    name="text-dense",
                    vector=dense_vector,
                ),
                limit=5,
                with_payload=True,
            ),
             models.SearchRequest(
                vector=models.NamedSparseVector(
                    name="text-sparse",
                    vector=sparse_vector,
                ),
                limit=5,
                with_payload=True,
            ),
        ],
    )

    # For simplicity, we're just returning the raw results from both searches.
    # In a real application you would likely want to rerank/combine these results.
    return {"dense_search_results": search_result[0], "sparse_search_results": search_result[1]}
