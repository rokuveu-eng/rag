#!/usr/bin/env python
# -*- coding: utf-8 -*-
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Query
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from qdrant_client import QdrantClient, models
import httpx
from rank_bm25 import BM25Okapi
import json
import openpyxl
import io
import logging
import os

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
CORPUS_DIR = "corpus_data"
os.makedirs(CORPUS_DIR, exist_ok=True)

def get_corpus_path(collection_name: str) -> str:
    return os.path.join(CORPUS_DIR, f"{collection_name}.json")

def save_corpus(collection_name: str, corpus: list):
    with open(get_corpus_path(collection_name), 'w', encoding='utf-8') as f:
        json.dump(corpus, f, ensure_ascii=False, indent=4)

def load_corpus(collection_name: str) -> list:
    corpus_path = get_corpus_path(collection_name)
    if os.path.exists(corpus_path):
        with open(corpus_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return []

def create_collection(collection_name: str):
    try:
        qdrant_client.get_collection(collection_name=collection_name)
    except Exception:
        qdrant_client.recreate_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(size=1024, distance=models.Distance.COSINE),
        )

async def get_embedding(text: str):
    async with httpx.AsyncClient(timeout=300.0) as client: 
        response = await client.post(ollama_api_url, json={"model": "bge-m3", "prompt": text})
        response.raise_for_status()
        return response.json()["embedding"]

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
        corpus = load_corpus(collection_name)
        
        rows_to_process = list(sheet.iter_rows(min_row=skip_rows + 2, values_only=True))
        points = []

        for i, row in enumerate(rows_to_process):
            text_to_embed = f"Артикул - {row[mappings['Артикул']]}, Наименование - {row[mappings['Наименование']]}, Тариф с НДС, руб - {row[mappings['Тариф с НДС, руб']]}, Имя файла - {file.filename}"
            documents.append(text_to_embed)
            corpus.append(text_to_embed) # Add to corpus for BM25

            embedding = await get_embedding(text_to_embed)
            payload = {}
            for header, col_idx in mappings.items():
                payload[header] = row[col_idx]
            payload["Остаток"] = ""

            points.append(models.PointStruct(
                id=len(corpus) - len(rows_to_process) + i, # Ensure unique IDs
                vector=embedding,
                payload=payload
            ))

        if points:
            qdrant_client.upsert(
                collection_name=collection_name,
                wait=True,
                points=points,
            )
        save_corpus(collection_name, corpus)
        return {"status": "success", "indexed_rows": len(documents)}
    except Exception as e:
        logger.error(f"Error processing file: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/search")
async def search(query: str, collection_name: str):
    corpus = load_corpus(collection_name)
    # Vector search
    embedding = await get_embedding(query)
    vector_search_result = qdrant_client.search(
        collection_name=collection_name,
        query_vector=embedding,
        limit=5,
    )

    # BM25 search
    bm25_docs = []
    if corpus:
        tokenized_corpus = [doc.split(" ") for doc in corpus]
        bm25 = BM25Okapi(tokenized_corpus)
        tokenized_query = query.split(" ")
        bm25_scores = bm25.get_scores(tokenized_query)
        
        # Combine results (simple approach: return top 5 from each)
        bm25_results = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:5]
        bm25_docs = [corpus[i] for i in bm25_results]

    return {"vector_search_results": vector_search_result, "bm25_search_results": bm25_docs}
