#!/usr/bin/env python
# -*- coding: utf-8 -*-
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Query, Body
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from qdrant_client import QdrantClient, models
from qdrant_client.http.exceptions import UnexpectedResponse
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
from time import perf_counter
from typing import Optional, List, Dict, Any
import uuid
import shutil
import time
from urllib.parse import urlencode
from pdf2image import convert_from_bytes
import pytesseract

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
upload_jobs = {}
stock_jobs = {}
passports_jobs = {}
chat_sessions: Dict[str, List[Dict[str, str]]] = {}
hf_home = os.getenv("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
bitrix_oauth_states: Dict[str, float] = {}

runtime_config: Dict[str, Dict[str, Any]] = {
    "polza": {
        "api_key": "",
        "model": "openai/gpt-4o",
        "base_url": "https://polza.ai/api/v1/chat/completions",
        "temperature": 0.2,
        "max_tokens": 500,
    },
    "bitrix": {
        "client_id": "",
        "client_secret": "",
        "redirect_uri": "",
        "webhook_url": "",
        "portal_base_url": "",
        "oauth_auth_url": "https://oauth.bitrix.info/oauth/authorize/",
        "oauth_token_url": "https://oauth.bitrix.info/oauth/token/",
        "access_token": "",
        "refresh_token": "",
        "token_expires_at": 0,
        "bot_id": "",
        "collection_name": "my_collection",
        "docs_collection_name": "passports_collection",
        "search_mode": "hybrid",
    },
}

def default_ollama_base_url():
    if os.path.exists("/.dockerenv"):
        return "http://ollama:11434"
    return "http://localhost:11434"

ollama_base_url = (os.getenv("OLLAMA_BASE_URL") or "").strip() or default_ollama_base_url()
ollama_api_url = f"{ollama_base_url}/api/embeddings"
ollama_batch_api_url = f"{ollama_base_url}/api/embed"
ollama_openai_embeddings_url = f"{ollama_base_url}/v1/embeddings"

# Initialize FastEmbed Sparse Model
# Using a standard SPLADE model which acts as a learned BM25 replacement.
sparse_embedding_model = SparseTextEmbedding(model_name="prithivida/Splade_PP_en_v1")


def mask_secret(value: str) -> str:
    if not value:
        return ""
    if len(value) <= 6:
        return "***"
    return f"{value[:3]}***{value[-2:]}"


def runtime_config_response() -> Dict[str, Any]:
    polza = runtime_config["polza"]
    bitrix = runtime_config["bitrix"]
    return {
        "polza": {
            "configured": bool(polza.get("api_key")),
            "api_key_masked": mask_secret(polza.get("api_key", "")),
            "model": polza.get("model"),
            "base_url": polza.get("base_url"),
            "temperature": polza.get("temperature"),
            "max_tokens": polza.get("max_tokens"),
        },
        "bitrix": {
            "client_id": bitrix.get("client_id", ""),
            "client_secret_masked": mask_secret(bitrix.get("client_secret", "")),
            "redirect_uri": bitrix.get("redirect_uri", ""),
            "webhook_url": bitrix.get("webhook_url", ""),
            "portal_base_url": bitrix.get("portal_base_url", ""),
            "oauth_auth_url": bitrix.get("oauth_auth_url", "https://oauth.bitrix.info/oauth/authorize/"),
            "oauth_token_url": bitrix.get("oauth_token_url", "https://oauth.bitrix.info/oauth/token/"),
            "oauth_connected": bool(bitrix.get("access_token")),
            "access_token_masked": mask_secret(bitrix.get("access_token", "")),
            "refresh_token_masked": mask_secret(bitrix.get("refresh_token", "")),
            "token_expires_at": bitrix.get("token_expires_at", 0),
            "bot_id": bitrix.get("bot_id", ""),
            "collection_name": bitrix.get("collection_name", "my_collection"),
            "docs_collection_name": bitrix.get("docs_collection_name", "passports_collection"),
            "search_mode": bitrix.get("search_mode", "hybrid"),
        },
    }


@app.get("/runtime_config")
async def get_runtime_config():
    return runtime_config_response()


@app.post("/runtime_config")
async def set_runtime_config(payload: Dict[str, Any] = Body(...)):
    polza = runtime_config["polza"]
    bitrix = runtime_config["bitrix"]

    polza_updates = payload.get("polza", {}) if isinstance(payload.get("polza", {}), dict) else {}
    bitrix_updates = payload.get("bitrix", {}) if isinstance(payload.get("bitrix", {}), dict) else {}

    if "api_key" in polza_updates:
        polza["api_key"] = str(polza_updates.get("api_key") or "").strip()
    if "model" in polza_updates:
        polza["model"] = str(polza_updates.get("model") or "openai/gpt-4o").strip() or "openai/gpt-4o"
    if "base_url" in polza_updates:
        polza["base_url"] = str(polza_updates.get("base_url") or "https://polza.ai/api/v1/chat/completions").strip()
    if "temperature" in polza_updates:
        try:
            temperature = float(polza_updates.get("temperature"))
            polza["temperature"] = max(0.0, min(2.0, temperature))
        except (TypeError, ValueError):
            raise HTTPException(status_code=400, detail="Invalid polza.temperature")
    if "max_tokens" in polza_updates:
        try:
            max_tokens = int(polza_updates.get("max_tokens"))
            if max_tokens <= 0:
                raise ValueError()
            polza["max_tokens"] = max_tokens
        except (TypeError, ValueError):
            raise HTTPException(status_code=400, detail="Invalid polza.max_tokens")

    for key in [
        "client_id",
        "client_secret",
        "redirect_uri",
        "webhook_url",
        "portal_base_url",
        "oauth_auth_url",
        "oauth_token_url",
        "bot_id",
        "collection_name",
        "docs_collection_name",
        "search_mode",
    ]:
        if key in bitrix_updates:
            bitrix[key] = str(bitrix_updates.get(key) or "").strip()

    if bool(bitrix_updates.get("clear_oauth")):
        bitrix["access_token"] = ""
        bitrix["refresh_token"] = ""
        bitrix["token_expires_at"] = 0

    mode = bitrix.get("search_mode", "hybrid").lower().strip()
    if mode not in {"hybrid", "dense", "sparse"}:
        raise HTTPException(status_code=400, detail="bitrix.search_mode must be hybrid, dense or sparse")
    bitrix["search_mode"] = mode

    return {"status": "updated", "config": runtime_config_response()}


def bitrix_rest_base_url() -> str:
    bitrix = runtime_config["bitrix"]
    portal_base_url = (bitrix.get("portal_base_url") or "").strip().rstrip("/")
    if not portal_base_url:
        return ""
    return f"{portal_base_url}/rest"


async def refresh_bitrix_access_token() -> bool:
    bitrix = runtime_config["bitrix"]
    refresh_token = (bitrix.get("refresh_token") or "").strip()
    client_id = (bitrix.get("client_id") or "").strip()
    client_secret = (bitrix.get("client_secret") or "").strip()
    token_url = (bitrix.get("oauth_token_url") or "https://oauth.bitrix.info/oauth/token/").strip()

    if not refresh_token or not client_id or not client_secret:
        return False

    payload = {
        "grant_type": "refresh_token",
        "client_id": client_id,
        "client_secret": client_secret,
        "refresh_token": refresh_token,
    }

    async with httpx.AsyncClient(timeout=20.0) as client:
        response = await client.post(token_url, data=payload)
        response.raise_for_status()
        data = response.json()

    access_token = str(data.get("access_token") or "").strip()
    new_refresh_token = str(data.get("refresh_token") or refresh_token).strip()
    expires_in = int(data.get("expires_in") or 3600)
    domain = str(data.get("domain") or "").strip()

    if not access_token:
        return False

    bitrix["access_token"] = access_token
    bitrix["refresh_token"] = new_refresh_token
    bitrix["token_expires_at"] = int(time.time()) + max(60, expires_in - 30)
    if domain:
        bitrix["portal_base_url"] = f"https://{domain}"
    return True


async def ensure_bitrix_access_token() -> bool:
    bitrix = runtime_config["bitrix"]
    token = (bitrix.get("access_token") or "").strip()
    expires_at = int(bitrix.get("token_expires_at") or 0)
    now = int(time.time())

    if token and expires_at > now + 30:
        return True

    try:
        return await refresh_bitrix_access_token()
    except Exception as exc:
        logger.error("Bitrix token refresh failed: %s", exc, exc_info=True)
        return False


async def bitrix_api_call(method: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    bitrix = runtime_config["bitrix"]
    webhook_url = (bitrix.get("webhook_url") or "").strip().rstrip("/")

    if webhook_url:
        method_url = f"{webhook_url}/{method}.json"
        async with httpx.AsyncClient(timeout=20.0) as client:
            response = await client.post(method_url, data=payload)
            response.raise_for_status()
            return response.json() if response.content else {"result": True}

    if not await ensure_bitrix_access_token():
        raise HTTPException(status_code=400, detail="Bitrix OAuth не подключён или токен недействителен")

    base_url = bitrix_rest_base_url()
    if not base_url:
        raise HTTPException(status_code=400, detail="Не задан bitrix.portal_base_url")

    method_url = f"{base_url}/{method}.json"
    oauth_payload = dict(payload)
    oauth_payload["auth"] = bitrix.get("access_token", "")

    async with httpx.AsyncClient(timeout=20.0) as client:
        response = await client.post(method_url, data=oauth_payload)
        if response.status_code == 401:
            refreshed = await refresh_bitrix_access_token()
            if not refreshed:
                raise HTTPException(status_code=401, detail="Не удалось обновить Bitrix OAuth токен")
            oauth_payload["auth"] = bitrix.get("access_token", "")
            response = await client.post(method_url, data=oauth_payload)
        response.raise_for_status()
        data = response.json() if response.content else {"result": True}

    if isinstance(data, dict) and data.get("error"):
        err = data.get("error_description") or data.get("error")
        raise HTTPException(status_code=400, detail=f"Bitrix API error: {err}")

    return data


@app.get("/bitrix/oauth/connect_url")
async def bitrix_oauth_connect_url():
    bitrix = runtime_config["bitrix"]
    client_id = (bitrix.get("client_id") or "").strip()
    redirect_uri = (bitrix.get("redirect_uri") or "").strip()
    auth_url = (bitrix.get("oauth_auth_url") or "https://oauth.bitrix.info/oauth/authorize/").strip()

    if not client_id or not redirect_uri:
        raise HTTPException(status_code=400, detail="Укажите bitrix.client_id и bitrix.redirect_uri")

    state = str(uuid.uuid4())
    bitrix_oauth_states[state] = time.time()
    query = urlencode(
        {
            "client_id": client_id,
            "response_type": "code",
            "redirect_uri": redirect_uri,
            "state": state,
        }
    )
    return {"connect_url": f"{auth_url}?{query}", "state": state}


@app.get("/bitrix/oauth/callback")
async def bitrix_oauth_callback(
    code: str = Query(...),
    state: str = Query(...),
    domain: Optional[str] = Query(None),
):
    state_created_at = bitrix_oauth_states.pop(state, None)
    if not state_created_at or (time.time() - state_created_at) > 900:
        raise HTTPException(status_code=400, detail="Недействительный OAuth state")

    bitrix = runtime_config["bitrix"]
    client_id = (bitrix.get("client_id") or "").strip()
    client_secret = (bitrix.get("client_secret") or "").strip()
    redirect_uri = (bitrix.get("redirect_uri") or "").strip()
    token_url = (bitrix.get("oauth_token_url") or "https://oauth.bitrix.info/oauth/token/").strip()

    if not client_id or not client_secret or not redirect_uri:
        raise HTTPException(status_code=400, detail="Заполните bitrix client_id/client_secret/redirect_uri")

    payload = {
        "grant_type": "authorization_code",
        "client_id": client_id,
        "client_secret": client_secret,
        "code": code,
        "redirect_uri": redirect_uri,
    }

    async with httpx.AsyncClient(timeout=20.0) as client:
        response = await client.post(token_url, data=payload)
        response.raise_for_status()
        data = response.json()

    access_token = str(data.get("access_token") or "").strip()
    refresh_token = str(data.get("refresh_token") or "").strip()
    expires_in = int(data.get("expires_in") or 3600)
    oauth_domain = str(data.get("domain") or domain or "").strip()

    if not access_token:
        raise HTTPException(status_code=400, detail="Bitrix OAuth не вернул access_token")

    bitrix["access_token"] = access_token
    bitrix["refresh_token"] = refresh_token
    bitrix["token_expires_at"] = int(time.time()) + max(60, expires_in - 30)
    if oauth_domain:
        bitrix["portal_base_url"] = f"https://{oauth_domain}"

    return {
        "status": "connected",
        "portal_base_url": bitrix.get("portal_base_url", ""),
        "expires_at": bitrix.get("token_expires_at", 0),
    }


@app.post("/bitrix/oauth/refresh")
async def bitrix_oauth_refresh():
    refreshed = await refresh_bitrix_access_token()
    if not refreshed:
        raise HTTPException(status_code=400, detail="Не удалось обновить OAuth токен")
    return {
        "status": "refreshed",
        "expires_at": runtime_config["bitrix"].get("token_expires_at", 0),
    }


@app.get("/bitrix/oauth/status")
async def bitrix_oauth_status():
    bitrix = runtime_config["bitrix"]
    now = int(time.time())
    expires_at = int(bitrix.get("token_expires_at") or 0)
    return {
        "connected": bool((bitrix.get("access_token") or "").strip()),
        "portal_base_url": bitrix.get("portal_base_url", ""),
        "expires_at": expires_at,
        "expires_in": max(0, expires_at - now),
        "access_token_masked": mask_secret(bitrix.get("access_token", "")),
        "refresh_token_masked": mask_secret(bitrix.get("refresh_token", "")),
    }

def safe_upsert(collection_name: str, points: list):
    try:
        qdrant_client.upsert(
            collection_name=collection_name,
            wait=True,
            points=points,
        )
        return
    except UnexpectedResponse as exc:
        message = str(exc)
        if "Payload error" in message and len(points) > 1:
            midpoint = len(points) // 2
            safe_upsert(collection_name, points[:midpoint])
            safe_upsert(collection_name, points[midpoint:])
            return
        raise


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


def ocr_pdf_bytes(contents: bytes) -> str:
    pages = convert_from_bytes(contents)
    extracted = []
    for page in pages:
        text = pytesseract.image_to_string(page, lang="rus+eng")
        if text:
            extracted.append(text.strip())
    return "\n".join(extracted).strip()


def chunk_text(text: str, *, max_chars: int = 2000, overlap: int = 200) -> List[str]:
    if not text:
        return []
    normalized = re.sub(r"\s+", " ", text).strip()
    if len(normalized) <= max_chars:
        return [normalized]
    chunks = []
    start = 0
    length = len(normalized)
    while start < length:
        end = min(start + max_chars, length)
        chunk = normalized[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= length:
            break
        start = max(0, end - overlap)
    return chunks


def build_passport_documents(file_entries: List[tuple]):
    documents = []
    payloads = []
    skipped = 0
    for filename, contents in file_entries:
        text = ocr_pdf_bytes(contents)
        if not text:
            skipped += 1
            continue
        chunks = chunk_text(text)
        if not chunks:
            skipped += 1
            continue
        for chunk_index, chunk in enumerate(chunks, start=1):
            documents.append(chunk)
            payloads.append(
                {
                    "pdf_name": filename,
                    "page_range": "1-2",
                    "source": "ocr",
                    "article": None,
                    "text": chunk,
                    "chunk_id": chunk_index,
                    "chunks_total": len(chunks),
                }
            )
    return documents, payloads, skipped


async def process_passports_upload(
    *,
    file_entries: List[tuple],
    collection_name: str,
    batch_size: int,
    points_batch_size: int,
    job_id: Optional[str] = None,
):
    if points_batch_size <= 0:
        raise HTTPException(status_code=400, detail="points_batch_size must be > 0")

    documents, payloads, skipped = build_passport_documents(file_entries)

    if not documents:
        raise HTTPException(status_code=400, detail="No text extracted from PDF files")

    total_start = perf_counter()
    indexed = 0
    total_chunks = len(documents)

    if job_id:
        passports_jobs[job_id].update(
            {
                "status": "running",
                "progress": 0,
                "indexed_chunks": 0,
                "total_chunks": total_chunks,
            }
        )

    for start in range(0, len(documents), batch_size):
        batch_docs = documents[start : start + batch_size]
        batch_payloads = payloads[start : start + batch_size]

        dense_task = asyncio.create_task(get_ollama_embeddings(batch_docs))
        sparse_vectors = list(sparse_embedding_model.embed(batch_docs))
        dense_vectors = await dense_task

        batch_points = []
        for offset, dense_vector in enumerate(dense_vectors):
            sparse_vector = sparse_vectors[offset]
            qdrant_sparse_vector = models.SparseVector(
                indices=sparse_vector.indices.tolist(),
                values=sparse_vector.values.tolist(),
            )
            batch_points.append(
                models.PointStruct(
                    id=str(uuid.uuid4()),
                    vector={
                        "text-dense": dense_vector,
                        "text-sparse": qdrant_sparse_vector,
                    },
                    payload=batch_payloads[offset],
                )
            )

        for chunk_start in range(0, len(batch_points), points_batch_size):
            chunk = batch_points[chunk_start : chunk_start + points_batch_size]
            safe_upsert(collection_name, chunk)
            indexed += len(chunk)

            if job_id and total_chunks:
                percent = (indexed / total_chunks) * 100
                passports_jobs[job_id].update(
                    {
                        "status": "running",
                        "progress": round(percent, 2),
                        "indexed_chunks": indexed,
                        "total_chunks": total_chunks,
                    }
                )

    duration = perf_counter() - total_start
    result = {
        "status": "success",
        "indexed_chunks": indexed,
        "skipped_files": skipped,
        "total_chunks": total_chunks,
        "duration_sec": round(duration, 3),
    }
    if job_id:
        passports_jobs[job_id].update(
            {
                "status": "completed",
                "progress": 100,
                "indexed_chunks": indexed,
                "total_chunks": total_chunks,
                "duration_sec": round(duration, 3),
            }
        )
    return result


@app.post("/upload_passports")
async def upload_passports(
    files: List[UploadFile] = File(...),
    collection_name: str = Form(...),
    batch_size: int = Form(8),
    points_batch_size: int = Form(200),
):
    if len(files) > 30:
        raise HTTPException(status_code=400, detail="Maximum 30 files per upload")

    create_collection(collection_name)
    file_entries = []
    for file in files:
        contents = await file.read()
        file_entries.append((file.filename, contents))
    return await process_passports_upload(
        file_entries=file_entries,
        collection_name=collection_name,
        batch_size=batch_size,
        points_batch_size=points_batch_size,
    )


async def run_passports_job(
    *,
    job_id: str,
    file_entries: List[tuple],
    collection_name: str,
    batch_size: int,
    points_batch_size: int,
):
    try:
        await process_passports_upload(
            file_entries=file_entries,
            collection_name=collection_name,
            batch_size=batch_size,
            points_batch_size=points_batch_size,
            job_id=job_id,
        )
    except Exception as exc:
        logger.error("Passport upload job failed: %s", exc, exc_info=True)
        passports_jobs[job_id].update({"status": "failed", "error": str(exc)})


@app.post("/upload_passports_async")
async def upload_passports_async(
    files: List[UploadFile] = File(...),
    collection_name: str = Form(...),
    batch_size: int = Form(8),
    points_batch_size: int = Form(200),
):
    if len(files) > 30:
        raise HTTPException(status_code=400, detail="Maximum 30 files per upload")

    create_collection(collection_name)
    file_entries = []
    for file in files:
        contents = await file.read()
        file_entries.append((file.filename, contents))

    job_id = str(uuid.uuid4())
    passports_jobs[job_id] = {"status": "queued", "progress": 0}
    asyncio.create_task(
        run_passports_job(
            job_id=job_id,
            file_entries=file_entries,
            collection_name=collection_name,
            batch_size=batch_size,
            points_batch_size=points_batch_size,
        )
    )
    return {"status": "started", "job_id": job_id}


@app.get("/passports_status/{job_id}")
async def passports_status(job_id: str):
    if job_id not in passports_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    return passports_jobs[job_id]


@app.get("/hf_cache/info")
async def hf_cache_info():
    cache_dir = os.getenv("HF_HUB_CACHE", os.path.join(hf_home, "hub"))
    total_bytes = 0
    file_count = 0
    if os.path.exists(cache_dir):
        for root, _, files in os.walk(cache_dir):
            for file in files:
                file_path = os.path.join(root, file)
                try:
                    total_bytes += os.path.getsize(file_path)
                    file_count += 1
                except OSError:
                    continue
    return {
        "hf_home": hf_home,
        "hf_hub_cache": cache_dir,
        "exists": os.path.exists(cache_dir),
        "file_count": file_count,
        "size_mb": round(total_bytes / (1024 * 1024), 2),
    }


@app.post("/hf_cache/clear")
async def hf_cache_clear():
    cache_dir = os.getenv("HF_HUB_CACHE", os.path.join(hf_home, "hub"))
    if not os.path.exists(cache_dir):
        return {
            "status": "already_empty",
            "hf_hub_cache": cache_dir,
        }

    for entry in os.listdir(cache_dir):
        path = os.path.join(cache_dir, entry)
        try:
            if os.path.isdir(path):
                shutil.rmtree(path)
            else:
                os.remove(path)
        except OSError as exc:
            logger.error("Failed to delete %s: %s", path, exc)
            raise HTTPException(status_code=500, detail=f"Failed to delete {path}: {exc}")

    return {
        "status": "cleared",
        "hf_hub_cache": cache_dir,
    }


@app.get("/search_passports")
async def search_passports(
    query: str = Query(...),
    collection_name: str = Query(...),
    limit: int = Query(5, ge=1, le=20),
):
    dense_vector = await get_ollama_embedding(query)
    sparse_vector_gen = list(sparse_embedding_model.embed([query]))[0]
    sparse_vector = models.SparseVector(
        indices=sparse_vector_gen.indices.tolist(),
        values=sparse_vector_gen.values.tolist(),
    )

    try:
        stage1_points = qdrant_client.query_points(
            collection_name=collection_name,
            prefetch=[
                models.Prefetch(
                    query=sparse_vector,
                    using="text-sparse",
                    limit=limit,
                ),
                models.Prefetch(
                    query=dense_vector,
                    using="text-dense",
                    limit=limit,
                ),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=limit,
            with_payload=True,
            with_vectors=True,
        ).points

        stage1_results = []
        pdf_counts = {}
        best_point = None
        best_score = None
        for point in stage1_points:
            payload = point.payload or {}
            stage1_results.append(
                {
                    "id": str(point.id),
                    "score": point.score,
                    "payload": payload,
                }
            )
            pdf_name = payload.get("pdf_name")
            if pdf_name:
                pdf_counts[pdf_name] = pdf_counts.get(pdf_name, 0) + 1
            if best_score is None or (point.score is not None and point.score > best_score):
                best_score = point.score
                best_point = point

        selected_pdf = (best_point.payload or {}).get("pdf_name") if best_point else None
        if not selected_pdf:
            return {
                "results": [],
                "stage1_results": stage1_results,
                "selected_pdf": None,
                "debug": {
                    "dense_dim": len(dense_vector),
                    "sparse_nonzero": len(sparse_vector.indices),
                    "stage1_hits": len(stage1_points),
                    "stage1_pdf_counts": pdf_counts,
                    "stage1_best_score": best_score,
                },
            }

        pdf_filter = models.Filter(
            must=[
                models.FieldCondition(
                    key="pdf_name",
                    match=models.MatchValue(value=selected_pdf),
                )
            ]
        )

        stage2_points = qdrant_client.query_points(
            collection_name=collection_name,
            prefetch=[
                models.Prefetch(
                    query=sparse_vector,
                    using="text-sparse",
                    limit=limit,
                ),
                models.Prefetch(
                    query=dense_vector,
                    using="text-dense",
                    limit=limit,
                ),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=limit,
            with_payload=True,
            with_vectors=True,
            query_filter=pdf_filter,
        ).points

        return {
            "results": stage2_points,
            "stage1_results": stage1_results,
            "selected_pdf": selected_pdf,
            "debug": {
                "dense_dim": len(dense_vector),
                "sparse_nonzero": len(sparse_vector.indices),
                "stage1_hits": len(stage1_points),
                "stage1_pdf_counts": pdf_counts,
                "stage1_best_score": best_score,
            },
        }
    except Exception as exc:
        logger.error("Passport search failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))


@app.delete("/collection")
async def delete_collection(collection_name: str = Query(...)):
    try:
        qdrant_client.delete_collection(collection_name=collection_name)
        return {"status": "deleted", "collection_name": collection_name}
    except Exception as exc:
        logger.error("Failed to delete collection %s: %s", collection_name, exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))

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
                    embeddings = data["embeddings"]
                    if len(embeddings) == len(texts):
                        return embeddings
                    logger.warning(
                        "Ollama /api/embed returned %s embeddings for %s inputs. Falling back.",
                        len(embeddings),
                        len(texts),
                    )
        except httpx.HTTPError:
            pass

        # Fallback: OpenAI-compatible embeddings endpoint (batch)
        try:
            openai_response = await client.post(
                ollama_openai_embeddings_url,
                json={"model": "bge-m3", "input": texts},
            )
            if openai_response.status_code < 400:
                data = openai_response.json()
                if "data" in data:
                    ordered = sorted(data["data"], key=lambda item: item.get("index", 0))
                    embeddings = [item["embedding"] for item in ordered]
                    if len(embeddings) == len(texts):
                        return embeddings
                    logger.warning(
                        "Ollama /v1/embeddings returned %s embeddings for %s inputs. Falling back.",
                        len(embeddings),
                        len(texts),
                    )
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
                if response.status_code == 404:
                    alt_response = await client.post(
                        ollama_openai_embeddings_url,
                        json={"model": "bge-m3", "input": text},
                    )
                    alt_response.raise_for_status()
                    data = alt_response.json()
                    return data["data"][0]["embedding"]
                response.raise_for_status()
                return response.json()["embedding"]

        try:
            return await asyncio.gather(*(fetch_one(text) for text in texts))
        except httpx.ConnectError as exc:
            logger.error(
                "Cannot connect to Ollama. Set OLLAMA_BASE_URL (e.g. http://ollama:11434 for Docker or http://localhost:11434 for local)."
            )
            raise exc

async def get_ollama_embedding(text: str):
    embeddings = await get_ollama_embeddings([text])
    return embeddings[0]

def get_sparse_embedding(text: str):
    # FastEmbed returns a generator of sparse embeddings
    embeddings = list(sparse_embedding_model.embed([text]))
    return embeddings[0]

async def process_xlsx_upload(
    *,
    contents: bytes,
    file_name: str,
    skip_rows: int,
    mappings: dict,
    collection_name: str,
    batch_size: int,
    points_batch_size: int,
    article_mode: str = "price",
    job_id: str = None,
    sheet_name: Optional[str] = None,
):
    if points_batch_size <= 0:
        raise HTTPException(status_code=400, detail="points_batch_size must be > 0")

    workbook = openpyxl.load_workbook(io.BytesIO(contents))
    if sheet_name:
        if sheet_name not in workbook.sheetnames:
            raise HTTPException(status_code=400, detail=f"Sheet '{sheet_name}' not found")
        sheet = workbook[sheet_name]
    else:
        sheet = workbook.active

    rows = list(sheet.iter_rows(min_row=skip_rows + 2, values_only=True))
    documents = []
    filtered_rows = []
    skipped_rows = 0

    for row in rows:
        # Safely handle None values in row
        def get_val(idx):
            if idx < len(row) and row[idx] is not None:
                return str(row[idx])
            return ""

        article_raw = get_val(mappings["Артикул"])
        name_raw = get_val(mappings["Наименование"])
        article_norm = normalize_article(article_raw)
        name_norm = str(name_raw).strip()

        if not article_norm and not name_norm:
            skipped_rows += 1
            continue

        text_to_embed = (
            f"Артикул - {article_raw}, "
            f"Наименование - {name_raw}, "
            f"Имя файла - {file_name}"
        )
        documents.append(text_to_embed)
        filtered_rows.append(row)

    rows = filtered_rows
    if skipped_rows:
        logger.info("Skipped %s rows without article and name", skipped_rows)

    total_start = perf_counter()
    indexed_rows = 0
    total_rows = len(documents)

    if job_id:
        upload_jobs[job_id].update(
            {
                "status": "running",
                "progress": 0,
                "indexed_rows": 0,
                "total_rows": total_rows,
                "rate": 0,
                "eta": None,
            }
        )

    # Batch dense embeddings via Ollama
    for start in range(0, len(documents), batch_size):
        batch_docs = documents[start : start + batch_size]
        batch_rows = rows[start : start + batch_size]

        embed_start = perf_counter()
        dense_task = asyncio.create_task(get_ollama_embeddings(batch_docs))
        sparse_vectors = list(sparse_embedding_model.embed(batch_docs))
        dense_vectors = await dense_task
        embed_duration = perf_counter() - embed_start

        batch_points = []
        for offset, dense_vector in enumerate(dense_vectors):
            i = start + offset
            sparse_vector = sparse_vectors[offset]

            qdrant_sparse_vector = models.SparseVector(
                indices=sparse_vector.indices.tolist(),
                values=sparse_vector.values.tolist(),
            )

            row_data = batch_rows[offset]
            payload = {}
            for header, col_idx in mappings.items():
                if col_idx < len(row_data):
                    payload[header] = row_data[col_idx]
            article_value = apply_article_mode(payload.get("Артикул"), article_mode)
            payload["Артикул"] = article_value
            payload["Остаток"] = 0
            payload["Имя файла"] = file_name
            point_id = build_point_id(collection_name, article_value)

            batch_points.append(
                models.PointStruct(
                    id=point_id,
                    vector={
                        "text-dense": dense_vector,
                        "text-sparse": qdrant_sparse_vector,
                    },
                    payload=payload,
                )
            )

        upsert_start = perf_counter()
        for chunk_start in range(0, len(batch_points), points_batch_size):
            chunk = batch_points[chunk_start : chunk_start + points_batch_size]
            safe_upsert(collection_name, chunk)
            indexed_rows += len(chunk)

            elapsed = perf_counter() - total_start
            rate = indexed_rows / elapsed if elapsed > 0 else 0
            remaining = total_rows - indexed_rows
            eta = remaining / rate if rate > 0 else 0
            percent = (indexed_rows / total_rows) * 100 if total_rows else 100

            logger.info(
                "Progress: %s/%s (%.1f%%), %.2f rows/sec, ETA %.1fs",
                indexed_rows,
                total_rows,
                percent,
                rate,
                eta,
            )

            if job_id:
                upload_jobs[job_id].update(
                    {
                        "status": "running",
                        "progress": round(percent, 2),
                        "indexed_rows": indexed_rows,
                        "total_rows": total_rows,
                        "rate": round(rate, 2),
                        "eta": round(eta, 1),
                    }
                )
        upsert_duration = perf_counter() - upsert_start

        logger.info(
            "Batch %s-%s: embeddings %.2fs, upsert %.2fs",
            start,
            start + len(batch_docs) - 1,
            embed_duration,
            upsert_duration,
        )

    total_duration = perf_counter() - total_start
    logger.info("Indexed %s rows in %.2fs", indexed_rows, total_duration)
    result = {
        "status": "success",
        "indexed_rows": indexed_rows,
        "duration_sec": round(total_duration, 3),
    }
    if job_id:
        upload_jobs[job_id].update(
            {
                "status": "completed",
                "progress": 100,
                "indexed_rows": indexed_rows,
                "total_rows": total_rows,
                "duration_sec": round(total_duration, 3),
            }
        )
    return result


@app.post("/upload_processed_xlsx")
async def upload_processed_xlsx(
    file: UploadFile = File(...),
    skip_rows: int = Form(...),
    mappings: str = Form(...),
    collection_name: str = Form(...),
    article_mode: str = Form("price"),
    batch_size: int = Form(16),
    points_batch_size: int = Form(200),
    sheet_name: Optional[str] = Form(None),
):
    logger.info(f"Received request to /upload_processed_xlsx for collection: {collection_name}")
    logger.info(
        f"skip_rows: {skip_rows}, mappings: {mappings}, batch_size: {batch_size}, points_batch_size: {points_batch_size}"
    )
    create_collection(collection_name)

    try:
        mappings = json.loads(mappings)
        if not mappings:
            raise HTTPException(status_code=400, detail="Mappings cannot be empty")
        logger.info(f"Parsed mappings: {mappings}")
        contents = await file.read()
        return await process_xlsx_upload(
            contents=contents,
            file_name=file.filename,
            skip_rows=skip_rows,
            mappings=mappings,
            collection_name=collection_name,
            batch_size=batch_size,
            points_batch_size=points_batch_size,
            article_mode=article_mode,
            sheet_name=sheet_name,
        )
    except Exception as e:
        logger.error(f"Error processing file: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


async def run_upload_job(
    *,
    job_id: str,
    contents: bytes,
    file_name: str,
    skip_rows: int,
    mappings: dict,
    collection_name: str,
    batch_size: int,
    points_batch_size: int,
    article_mode: str,
    sheet_name: Optional[str],
):
    try:
        await process_xlsx_upload(
            contents=contents,
            file_name=file_name,
            skip_rows=skip_rows,
            mappings=mappings,
            collection_name=collection_name,
            batch_size=batch_size,
            points_batch_size=points_batch_size,
            article_mode=article_mode,
            job_id=job_id,
            sheet_name=sheet_name,
        )
    except Exception as exc:
        logger.error("Async upload job failed: %s", exc, exc_info=True)
        upload_jobs[job_id].update({"status": "failed", "error": str(exc)})


@app.post("/upload_processed_xlsx_async")
async def upload_processed_xlsx_async(
    file: UploadFile = File(...),
    skip_rows: int = Form(...),
    mappings: str = Form(...),
    collection_name: str = Form(...),
    article_mode: str = Form("price"),
    batch_size: int = Form(16),
    points_batch_size: int = Form(200),
    sheet_name: Optional[str] = Form(None),
):
    logger.info(f"Received request to /upload_processed_xlsx_async for collection: {collection_name}")
    logger.info(
        f"skip_rows: {skip_rows}, mappings: {mappings}, batch_size: {batch_size}, points_batch_size: {points_batch_size}"
    )
    create_collection(collection_name)

    try:
        mappings = json.loads(mappings)
        if not mappings:
            raise HTTPException(status_code=400, detail="Mappings cannot be empty")
        logger.info(f"Parsed mappings: {mappings}")
        contents = await file.read()
    except Exception as e:
        logger.error(f"Error processing file: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

    job_id = str(uuid.uuid4())
    upload_jobs[job_id] = {"status": "queued", "progress": 0}
    asyncio.create_task(
        run_upload_job(
            job_id=job_id,
            contents=contents,
            file_name=file.filename,
            skip_rows=skip_rows,
            mappings=mappings,
            collection_name=collection_name,
            batch_size=batch_size,
            points_batch_size=points_batch_size,
            article_mode=article_mode,
            sheet_name=sheet_name,
        )
    )
    return {"status": "started", "job_id": job_id}


@app.get("/upload_status/{job_id}")
async def upload_status(job_id: str):
    if job_id not in upload_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    return upload_jobs[job_id]


def normalize_article(value) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if text.startswith('="') and text.endswith('"'):
        text = text[2:-1]
    if text.startswith("=") and text.endswith('"'):
        text = text[1:-1]
    if text.startswith('"') and text.endswith('"'):
        text = text[1:-1]
    return text.strip()


def apply_article_mode(article_value, article_mode: str) -> str:
    normalized = normalize_article(article_value)
    if not normalized:
        return normalized
    if article_mode == "chint":
        return f"{normalized}CHINT"
    if article_mode == "dkc":
        return f"DKC{normalized}"
    return normalized


def build_point_id(collection_name: str, article_value: str) -> str:
    if not article_value:
        return str(uuid.uuid4())
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"{collection_name}:{article_value}"))


def reset_stock_payload(collection_name: str):
    qdrant_client.set_payload(
        collection_name=collection_name,
        payload={"Остаток": 0},
        points=models.Filter(must=[]),
    )


async def process_stock_upload(
    *,
    contents: bytes,
    skip_rows: int,
    article_col: int,
    stock_col: int,
    collection_name: str,
    job_id: str = None,
    batch_size: int = 200,
):
    workbook = openpyxl.load_workbook(io.BytesIO(contents))
    sheet = workbook.active

    rows = list(sheet.iter_rows(min_row=skip_rows + 2, values_only=True))
    total_rows = len(rows)
    processed = 0
    updated = 0
    skipped = 0
    total_start = perf_counter()

    if job_id:
        stock_jobs[job_id].update(
            {
                "status": "running",
                "progress": 0,
                "processed_rows": 0,
                "updated_rows": 0,
                "skipped_rows": 0,
                "total_rows": total_rows,
                "rate": 0,
                "eta": None,
            }
        )

    for start in range(0, total_rows, batch_size):
        batch_rows = rows[start : start + batch_size]
        batch_articles = []
        batch_payloads = {}

        for row in batch_rows:
            article = row[article_col] if article_col < len(row) else None
            stock_value = row[stock_col] if stock_col < len(row) else None
            if article is None:
                skipped += 1
                continue
            article_str = normalize_article(article)
            if not article_str:
                skipped += 1
                continue
            batch_articles.append(article_str)
            batch_payloads[article_str] = stock_value

        if not batch_articles:
            processed += len(batch_rows)
            continue

        matched_points = qdrant_client.scroll(
            collection_name=collection_name,
            scroll_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="Артикул",
                        match=models.MatchAny(any=batch_articles),
                    )
                ]
            ),
            limit=len(batch_articles),
            with_payload=True,
            with_vectors=False,
        )[0]

        found_articles = set()
        for point in matched_points:
            article_value = point.payload.get("Артикул")
            if article_value is None:
                continue
            article_str = normalize_article(article_value)
            found_articles.add(article_str)
            if article_str not in batch_payloads:
                continue
            qdrant_client.set_payload(
                collection_name=collection_name,
                payload={"Остаток": batch_payloads.get(article_str)},
                points=[point.id],
            )
            updated += 1

        missing_articles = set(batch_articles) - found_articles
        skipped += len(missing_articles)

        processed += len(batch_rows)
        elapsed = perf_counter() - total_start
        rate = processed / elapsed if elapsed > 0 else 0
        remaining = total_rows - processed
        eta = remaining / rate if rate > 0 else 0
        percent = (processed / total_rows) * 100 if total_rows else 100

        logger.info(
            "Stock progress: %s/%s (%.1f%%), %.2f rows/sec, ETA %.1fs",
            processed,
            total_rows,
            percent,
            rate,
            eta,
        )

        if job_id:
            stock_jobs[job_id].update(
                {
                    "status": "running",
                    "progress": round(percent, 2),
                    "processed_rows": processed,
                    "updated_rows": updated,
                    "skipped_rows": skipped,
                    "total_rows": total_rows,
                    "rate": round(rate, 2),
                    "eta": round(eta, 1),
                }
            )

    duration = perf_counter() - total_start
    result = {
        "status": "success",
        "processed_rows": processed,
        "updated_rows": updated,
        "skipped_rows": skipped,
        "duration_sec": round(duration, 3),
    }

    if job_id:
        stock_jobs[job_id].update(
            {
                "status": "completed",
                "progress": 100,
                "processed_rows": processed,
                "updated_rows": updated,
                "skipped_rows": skipped,
                "duration_sec": round(duration, 3),
            }
        )

    return result


async def run_stock_job(
    *,
    job_id: str,
    contents: bytes,
    skip_rows: int,
    article_col: int,
    stock_col: int,
    collection_name: str,
    batch_size: int,
):
    try:
        stock_jobs[job_id].update({"status": "resetting", "progress": 0})
        reset_stock_payload(collection_name)
        await process_stock_upload(
            contents=contents,
            skip_rows=skip_rows,
            article_col=article_col,
            stock_col=stock_col,
            collection_name=collection_name,
            job_id=job_id,
            batch_size=batch_size,
        )
    except Exception as exc:
        logger.error("Stock upload job failed: %s", exc, exc_info=True)
        stock_jobs[job_id].update({"status": "failed", "error": str(exc)})


@app.post("/upload_stock_async")
async def upload_stock_async(
    file: UploadFile = File(...),
    skip_rows: int = Form(...),
    article_col: int = Form(...),
    stock_col: int = Form(...),
    collection_name: str = Form(...),
    batch_size: int = Form(200),
):
    logger.info(f"Received request to /upload_stock_async for collection: {collection_name}")
    logger.info(
        f"skip_rows: {skip_rows}, article_col: {article_col}, stock_col: {stock_col}, batch_size: {batch_size}"
    )

    try:
        contents = await file.read()
    except Exception as e:
        logger.error(f"Error reading stock file: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

    job_id = str(uuid.uuid4())
    stock_jobs[job_id] = {"status": "queued", "progress": 0}
    asyncio.create_task(
        run_stock_job(
            job_id=job_id,
            contents=contents,
            skip_rows=skip_rows,
            article_col=article_col,
            stock_col=stock_col,
            collection_name=collection_name,
            batch_size=batch_size,
        )
    )
    return {"status": "started", "job_id": job_id}


@app.get("/stock_status/{job_id}")
async def stock_status(job_id: str):
    if job_id not in stock_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    return stock_jobs[job_id]


async def run_catalog_search(
    *,
    query: str,
    collection_name: str,
    only_in_stock: bool = False,
    mode: str = "hybrid",
):
    mode = mode.lower().strip()
    if mode not in {"hybrid", "dense", "sparse"}:
        raise HTTPException(status_code=400, detail="mode must be hybrid, dense or sparse")

    dense_vector = None
    sparse_vector = None

    if mode in {"hybrid", "dense"}:
        dense_vector = await get_ollama_embedding(query)

    if mode in {"hybrid", "sparse"}:
        sparse_vector_gen = list(sparse_embedding_model.embed([query]))[0]
        sparse_vector = models.SparseVector(
            indices=sparse_vector_gen.indices.tolist(),
            values=sparse_vector_gen.values.tolist(),
        )

    def query_points(limit: int, query_filter: Optional[models.Filter]):
        if mode == "dense":
            return qdrant_client.query_points(
                collection_name=collection_name,
                query=dense_vector,
                using="text-dense",
                limit=limit,
                with_payload=True,
                query_filter=query_filter,
            ).points
        if mode == "sparse":
            return qdrant_client.query_points(
                collection_name=collection_name,
                query=sparse_vector,
                using="text-sparse",
                limit=limit,
                with_payload=True,
                query_filter=query_filter,
            ).points
        return qdrant_client.query_points(
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
            limit=limit,
            with_payload=True,
            query_filter=query_filter,
        ).points

    in_stock_filter = models.Filter(
        must=[
            models.FieldCondition(
                key="Остаток",
                range=models.Range(gt=0),
            )
        ]
    )

    if only_in_stock:
        points = query_points(15, in_stock_filter)
        return {
            "results": points,
            "debug": {
                "mode": mode,
                "dense_dim": len(dense_vector) if dense_vector is not None else None,
                "sparse_nonzero": len(sparse_vector.indices) if sparse_vector is not None else None,
            },
        }

    in_stock_points = query_points(5, in_stock_filter)
    general_points = query_points(15, None)

    seen_ids = {point.id for point in in_stock_points}
    combined = list(in_stock_points)
    for point in general_points:
        if point.id in seen_ids:
            continue
        combined.append(point)
        seen_ids.add(point.id)
        if len(combined) >= 15:
            break

    return {
        "results": combined,
        "debug": {
            "mode": mode,
            "dense_dim": len(dense_vector) if dense_vector is not None else None,
            "sparse_nonzero": len(sparse_vector.indices) if sparse_vector is not None else None,
        },
    }


def build_context_from_results(results: List[Any], limit: int = 5) -> str:
    lines = []
    for idx, point in enumerate(results[:limit], start=1):
        payload = point.payload or {}
        article = payload.get("Артикул", "")
        name = payload.get("Наименование", "")
        price = payload.get("Тариф с НДС, руб", "")
        stock = payload.get("Остаток", "")
        lines.append(
            f"{idx}. Артикул: {article}; Наименование: {name}; Цена: {price}; Остаток: {stock}"
        )
    return "\n".join(lines)


async def polza_chat_completion(dialog_id: str, user_query: str, context: str) -> str:
    polza = runtime_config["polza"]
    api_key = polza.get("api_key", "").strip()
    if not api_key:
        return "LLM не настроена: укажите Polza API key в веб-интерфейсе."

    history = chat_sessions.setdefault(dialog_id, [])
    system_prompt = (
        "Ты помощник по прайс-листу. Отвечай кратко и по фактам. "
        "Если данных недостаточно — честно сообщи об этом."
    )

    if user_query.startswith("[docs]"):
        system_prompt = (
            "Ты помощник по внутренней документации. Отвечай только на основе найденного контекста. "
            "Если данных в контексте недостаточно — так и напиши."
        )
    elif user_query.startswith("[stock]"):
        system_prompt = (
            "Ты помощник по остаткам. Отвечай кратко, с акцентом на наличие и артикулы. "
            "Не придумывай данные, используй только контекст."
        )

    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(history[-10:])
    messages.append(
        {
            "role": "user",
            "content": f"Запрос: {user_query}\n\nДоступные данные поиска:\n{context}",
        }
    )

    body = {
        "model": polza.get("model") or "openai/gpt-4o",
        "messages": messages,
        "temperature": polza.get("temperature", 0.2),
        "max_tokens": polza.get("max_tokens", 500),
    }

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                polza.get("base_url") or "https://polza.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json=body,
            )
            response.raise_for_status()
            data = response.json()
            content = (
                data.get("choices", [{}])[0]
                .get("message", {})
                .get("content", "")
                .strip()
            )
            if not content:
                content = "Не удалось получить текстовый ответ от LLM."
    except Exception as exc:
        logger.error("Polza request failed: %s", exc, exc_info=True)
        content = "LLM временно недоступна. Ниже — найденные позиции без генерации."

    history.append({"role": "user", "content": user_query})
    history.append({"role": "assistant", "content": content})
    if len(history) > 20:
        del history[:-20]
    return content


def extract_bitrix_message(payload: Dict[str, Any]) -> Dict[str, str]:
    data = payload.get("data", {}) if isinstance(payload.get("data"), dict) else {}
    params = data.get("PARAMS", {}) if isinstance(data.get("PARAMS"), dict) else {}

    text = (
        data.get("MESSAGE")
        or params.get("MESSAGE")
        or payload.get("message")
        or ""
    )
    text = str(text).strip()

    dialog_id = (
        data.get("DIALOG_ID")
        or params.get("DIALOG_ID")
        or data.get("CHAT_ID")
        or params.get("CHAT_ID")
        or payload.get("dialog_id")
        or payload.get("chat_id")
        or "default-dialog"
    )

    user_id = (
        data.get("FROM_USER_ID")
        or params.get("FROM_USER_ID")
        or payload.get("user_id")
        or ""
    )

    return {
        "text": text,
        "dialog_id": str(dialog_id),
        "user_id": str(user_id),
    }


async def send_bitrix_message(dialog_id: str, message: str):
    payload: Dict[str, Any] = {
        "DIALOG_ID": dialog_id,
        "MESSAGE": message,
    }
    bot_id = runtime_config["bitrix"].get("bot_id", "").strip()
    if bot_id:
        payload["BOT_ID"] = bot_id

    try:
        await bitrix_api_call("imbot.message.add", payload)
    except Exception as exc:
        logger.error("Failed to send Bitrix message: %s", exc, exc_info=True)


def extract_bot_id_from_bitrix_result(result_payload: Dict[str, Any]) -> str:
    if not isinstance(result_payload, dict):
        return ""
    result = result_payload.get("result")
    if isinstance(result, (int, str)):
        return str(result)
    if isinstance(result, dict):
        for key in ("BOT_ID", "bot_id", "ID", "id"):
            value = result.get(key)
            if value not in (None, ""):
                return str(value)
    return ""


@app.post("/bitrix/bot/register")
async def bitrix_bot_register(payload: Dict[str, Any] = Body(default={})):  # noqa: B008
    bitrix = runtime_config["bitrix"]
    handler_url = str(payload.get("handler_url") or "").strip()
    if not handler_url:
        redirect_uri = str(bitrix.get("redirect_uri") or "").strip()
        if redirect_uri.startswith("http"):
            handler_url = redirect_uri.rsplit("/bitrix/oauth/callback", 1)[0] + "/bitrix/webhook"

    if not handler_url:
        raise HTTPException(status_code=400, detail="Укажите handler_url (путь обработчика)")

    properties: Dict[str, Any] = {
        "NAME": str(payload.get("name") or "Bot").strip() or "Bot",
        "LAST_NAME": str(payload.get("last_name") or "").strip(),
        "COLOR": str(payload.get("color") or "AQUA").strip() or "AQUA",
        "EMAIL": str(payload.get("email") or "").strip(),
        "WORK_POSITION": str(payload.get("work_position") or "").strip(),
    }
    properties = {k: v for k, v in properties.items() if v not in (None, "")}

    register_payload: Dict[str, Any] = {
        "CODE": str(payload.get("code") or f"bot_{uuid.uuid4().hex[:8]}").strip(),
        "TYPE": str(payload.get("type") or "B").strip() or "B",
        "EVENT_MESSAGE_ADD": handler_url,
        "EVENT_WELCOME_MESSAGE": handler_url,
    }
    if properties:
        register_payload["PROPERTIES"] = properties

    api_result = await bitrix_api_call("imbot.register", register_payload)
    bot_id = extract_bot_id_from_bitrix_result(api_result)
    if bot_id:
        bitrix["bot_id"] = bot_id

    return {
        "status": "registered",
        "bot_id": bot_id,
        "handler_url": handler_url,
        "bitrix_result": api_result,
    }


@app.post("/bitrix/bot/update")
async def bitrix_bot_update(payload: Dict[str, Any] = Body(default={})):  # noqa: B008
    bitrix = runtime_config["bitrix"]
    bot_id = str(payload.get("bot_id") or bitrix.get("bot_id") or "").strip()
    if not bot_id:
        raise HTTPException(status_code=400, detail="Укажите bot_id для обновления")

    update_payload: Dict[str, Any] = {
        "BOT_ID": bot_id,
    }
    if payload.get("name"):
        update_payload["NAME"] = str(payload.get("name") or "").strip()
    if payload.get("last_name"):
        update_payload["LAST_NAME"] = str(payload.get("last_name") or "").strip()
    if payload.get("color"):
        update_payload["COLOR"] = str(payload.get("color") or "").strip()
    if payload.get("work_position"):
        update_payload["WORK_POSITION"] = str(payload.get("work_position") or "").strip()

    api_result = await bitrix_api_call("imbot.update", update_payload)
    bitrix["bot_id"] = bot_id
    return {
        "status": "updated",
        "bot_id": bot_id,
        "bitrix_result": api_result,
    }


async def handle_bot_command(dialog_id: str, text: str, *, send_to_bitrix: bool) -> Dict[str, Any]:
    async def emit(message: str):
        if send_to_bitrix:
            await send_bitrix_message(dialog_id, message)

    normalized = text.strip()
    lower = normalized.lower()

    if lower in {"/help", "help"}:
        response_text = (
            "Команды бота:\n"
            "/help — помощь\n"
            "/search <запрос> — поиск по документации + LLM\n"
            "/price <запрос> — поиск по прайс-листу + LLM\n"
            "/stock <запрос> — поиск только в наличии\n"
            "/newchat — очистить контекст диалога\n"
            "/clear — очистить контекст диалога"
        )
        await emit(response_text)
        return {"status": "ok", "reply": response_text}

    if lower in {"/newchat", "/clear"}:
        chat_sessions[dialog_id] = []
        response_text = "Контекст диалога очищен. Начинаем новый чат."
        await emit(response_text)
        return {"status": "ok", "reply": response_text}

    if lower.startswith("/price"):
        query_text = normalized[6:].strip()
        if not query_text:
            response_text = "Укажи запрос после команды: /price <что ищем>"
            await emit(response_text)
            return {"status": "ok", "reply": response_text}

        bitrix_cfg = runtime_config["bitrix"]
        collection_name = bitrix_cfg.get("collection_name", "my_collection") or "my_collection"
        mode = bitrix_cfg.get("search_mode", "hybrid") or "hybrid"

        try:
            search_payload = await run_catalog_search(
                query=query_text,
                collection_name=collection_name,
                only_in_stock=False,
                mode=mode,
            )
            results = search_payload.get("results", [])
            if not results:
                response_text = "Ничего не найдено по прайс-листу."
                await emit(response_text)
                return {"status": "ok", "reply": response_text}

            context = build_context_from_results(results, limit=5)
            llm_answer = await polza_chat_completion(dialog_id, query_text, context)
            response_text = f"{llm_answer}\n\nНайдено:\n{context}"
            await emit(response_text[:3800])
            return {"status": "ok", "reply": response_text}
        except Exception as exc:
            logger.error("/price processing failed: %s", exc, exc_info=True)
            response_text = f"Ошибка обработки /price: {exc}"
            await emit(response_text)
            return {"status": "error", "reply": response_text}

    if lower.startswith("/search"):
        query_text = normalized[7:].strip()
        if not query_text:
            response_text = "Укажи запрос после команды: /search <что ищем в документации>"
            await emit(response_text)
            return {"status": "ok", "reply": response_text}

        bitrix_cfg = runtime_config["bitrix"]
        docs_collection_name = bitrix_cfg.get("docs_collection_name", "passports_collection") or "passports_collection"
        mode = bitrix_cfg.get("search_mode", "hybrid") or "hybrid"

        try:
            search_payload = await run_catalog_search(
                query=query_text,
                collection_name=docs_collection_name,
                only_in_stock=False,
                mode=mode,
            )
            results = search_payload.get("results", [])
            if not results:
                response_text = "По документации ничего не найдено."
                await emit(response_text)
                return {"status": "ok", "reply": response_text}

            context = build_docs_context(results, limit=5)
            llm_answer = await polza_chat_completion(dialog_id, f"[docs] {query_text}", context)
            response_text = f"{llm_answer}\n\nИсточники:\n{context}"
            await emit(response_text[:3800])
            return {"status": "ok", "reply": response_text}
        except Exception as exc:
            logger.error("/search processing failed: %s", exc, exc_info=True)
            response_text = f"Ошибка обработки /search: {exc}"
            await emit(response_text)
            return {"status": "error", "reply": response_text}

    if lower.startswith("/stock"):
        query_text = normalized[6:].strip()
        if not query_text:
            response_text = "Укажи запрос после команды: /stock <артикул или наименование>"
            await emit(response_text)
            return {"status": "ok", "reply": response_text}

        bitrix_cfg = runtime_config["bitrix"]
        collection_name = bitrix_cfg.get("collection_name", "my_collection") or "my_collection"
        mode = bitrix_cfg.get("search_mode", "hybrid") or "hybrid"

        try:
            search_payload = await run_catalog_search(
                query=query_text,
                collection_name=collection_name,
                only_in_stock=True,
                mode=mode,
            )
            results = search_payload.get("results", [])
            if not results:
                response_text = "По наличию ничего не найдено."
                await emit(response_text)
                return {"status": "ok", "reply": response_text}

            context = build_context_from_results(results, limit=5)
            llm_answer = await polza_chat_completion(dialog_id, f"[stock] {query_text}", context)
            response_text = f"{llm_answer}\n\nВ наличии:\n{context}"
            await emit(response_text[:3800])
            return {"status": "ok", "reply": response_text}
        except Exception as exc:
            logger.error("/stock processing failed: %s", exc, exc_info=True)
            response_text = f"Ошибка обработки /stock: {exc}"
            await emit(response_text)
            return {"status": "error", "reply": response_text}

    response_text = "Неизвестная команда. Используй /help"
    await emit(response_text)
    return {"status": "ok", "reply": response_text}


@app.post("/bitrix/test_chat")
async def bitrix_test_chat(payload: Dict[str, Any] = Body(default={})):  # noqa: B008
    dialog_id = str(payload.get("dialog_id") or "web-test-dialog").strip() or "web-test-dialog"
    message = str(payload.get("message") or "").strip()
    if not message:
        raise HTTPException(status_code=400, detail="Укажите message")
    return await handle_bot_command(dialog_id, message, send_to_bitrix=False)


def build_docs_context(results: List[Any], limit: int = 5) -> str:
    lines = []
    for idx, point in enumerate(results[:limit], start=1):
        payload = point.payload or {}
        source = payload.get("pdf_name") or payload.get("source") or "doc"
        text = str(payload.get("text") or "").strip()
        preview = (text[:300] + "...") if len(text) > 300 else text
        lines.append(f"{idx}. Источник: {source}; Фрагмент: {preview}")
    return "\n".join(lines)


@app.post("/bitrix/webhook")
async def bitrix_webhook(payload: Dict[str, Any] = Body(default={})):  # noqa: B008
    incoming = extract_bitrix_message(payload if isinstance(payload, dict) else {})
    text = incoming["text"]
    dialog_id = incoming["dialog_id"]

    if not text:
        return {"status": "ignored", "reason": "empty message"}
    return await handle_bot_command(dialog_id, text, send_to_bitrix=True)

@app.get("/search")
async def search(
    query: str = Query(...),
    collection_name: str = Query(...),
    only_in_stock: bool = Query(False),
    mode: str = Query("hybrid"),
):
    try:
        return await run_catalog_search(
            query=query,
            collection_name=collection_name,
            only_in_stock=only_in_stock,
            mode=mode,
        )
    except Exception as e:
        logger.error(f"Search failed: {e}", exc_info=True)
        # Return error details to the client for easier debugging
        raise HTTPException(status_code=500, detail=str(e))
