#!/usr/bin/env python
# -*- coding: utf-8 -*-
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Query
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from qdrant_client import QdrantClient, models
from qdrant_client.http.exceptions import UnexpectedResponse
from fastembed import SparseTextEmbedding
import httpx
import json
import openpyxl
import io
import logging
import re
import asyncio
import os
import importlib.metadata
from time import perf_counter
from typing import Optional, List, Dict, Any
import uuid
from pdf2image import convert_from_bytes
import pytesseract
import math


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

logger.info("Starting API...")
for pkg in ("qdrant-client", "fastembed", "fastapi"):
    try:
        logger.info("%s version: %s", pkg, importlib.metadata.version(pkg))
    except Exception:
        logger.info("%s version: unknown", pkg)

app = FastAPI(title="Hybrid Search API")
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/", response_class=HTMLResponse)
async def read_root():
    with open("static/index.html") as f:
        return f.read()


qdrant_client = QdrantClient(
    host=os.getenv("QDRANT_HOST", "qdrant"),
    port=int(os.getenv("QDRANT_PORT", "6333")),
)

upload_jobs: Dict[str, Dict[str, Any]] = {}
stock_jobs: Dict[str, Dict[str, Any]] = {}
passports_jobs: Dict[str, Dict[str, Any]] = {}


def default_ollama_base_url():
    if os.path.exists("/.dockerenv"):
        return "http://ollama:11434"
    return "http://localhost:11434"


ollama_base_url = (os.getenv("OLLAMA_BASE_URL") or "").strip() or default_ollama_base_url()
ollama_api_url = f"{ollama_base_url}/api/embeddings"
ollama_batch_api_url = f"{ollama_base_url}/api/embed"
ollama_openai_embeddings_url = f"{ollama_base_url}/v1/embeddings"


sparse_embedding_model = SparseTextEmbedding(model_name="Qdrant/bm25")


def safe_upsert(collection_name: str, points: list):
    try:
        qdrant_client.upsert(collection_name=collection_name, wait=True, points=points)
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
                    index=models.SparseIndexParams(on_disk=False)
                )
            },
        )


async def get_ollama_embeddings(texts, concurrency=4):
    async with httpx.AsyncClient(timeout=300.0) as client:
        try:
            batch_response = await client.post(
                ollama_batch_api_url,
                json={"model": "bge-m3", "input": texts},
            )
            if batch_response.status_code < 400:
                data = batch_response.json()
                if "embeddings" in data and len(data["embeddings"]) == len(texts):
                    return data["embeddings"]
        except httpx.HTTPError:
            pass

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
        except httpx.HTTPError:
            pass

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

        return await asyncio.gather(*(fetch_one(text) for text in texts))


async def get_ollama_embedding(text: str):
    embeddings = await get_ollama_embeddings([text])
    return embeddings[0]


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


def cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
    if not vec_a or not vec_b or len(vec_a) != len(vec_b):
        return 0.0
    dot = 0.0
    norm_a = 0.0
    norm_b = 0.0
    for a, b in zip(vec_a, vec_b):
        dot += a * b
        norm_a += a * a
        norm_b += b * b
    if norm_a <= 0 or norm_b <= 0:
        return 0.0
    return dot / (math.sqrt(norm_a) * math.sqrt(norm_b))


def candidate_text_from_payload(payload: Dict[str, Any]) -> str:
    article = str(payload.get("Артикул") or "").strip()
    name = str(payload.get("Наименование") or "").strip()
    if article and name:
        return f"{article} {name}"
    return article or name or str(payload)


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
    sheet = workbook[sheet_name] if sheet_name else workbook.active

    rows = list(sheet.iter_rows(min_row=skip_rows + 2, values_only=True))
    documents = []
    filtered_rows = []

    for row in rows:
        def get_val(idx):
            if idx < len(row) and row[idx] is not None:
                return str(row[idx])
            return ""

        article_raw = get_val(mappings["Артикул"])
        name_raw = get_val(mappings["Наименование"])
        article_norm = normalize_article(article_raw)
        name_norm = str(name_raw).strip()
        if not article_norm and not name_norm:
            continue

        documents.append(
            f"Артикул - {article_raw}, Наименование - {name_raw}, Имя файла - {file_name}"
        )
        filtered_rows.append(row)

    rows = filtered_rows
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

    for start in range(0, len(documents), batch_size):
        batch_docs = documents[start : start + batch_size]
        batch_rows = rows[start : start + batch_size]

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

        for chunk_start in range(0, len(batch_points), points_batch_size):
            chunk = batch_points[chunk_start : chunk_start + points_batch_size]
            safe_upsert(collection_name, chunk)
            indexed_rows += len(chunk)

            elapsed = perf_counter() - total_start
            rate = indexed_rows / elapsed if elapsed > 0 else 0
            remaining = total_rows - indexed_rows
            eta = remaining / rate if rate > 0 else 0
            percent = (indexed_rows / total_rows) * 100 if total_rows else 100

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

    total_duration = perf_counter() - total_start
    result = {"status": "success", "indexed_rows": indexed_rows, "duration_sec": round(total_duration, 3)}
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


async def run_upload_job(**kwargs):
    job_id = kwargs["job_id"]
    try:
        await process_xlsx_upload(**kwargs)
    except Exception as exc:
        logger.error("Async upload job failed: %s", exc, exc_info=True)
        upload_jobs[job_id].update({"status": "failed", "error": str(exc)})


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
    create_collection(collection_name)
    mappings_obj = json.loads(mappings)
    contents = await file.read()
    return await process_xlsx_upload(
        contents=contents,
        file_name=file.filename,
        skip_rows=skip_rows,
        mappings=mappings_obj,
        collection_name=collection_name,
        batch_size=batch_size,
        points_batch_size=points_batch_size,
        article_mode=article_mode,
        sheet_name=sheet_name,
    )


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
    create_collection(collection_name)
    mappings_obj = json.loads(mappings)
    contents = await file.read()

    job_id = str(uuid.uuid4())
    upload_jobs[job_id] = {"status": "queued", "progress": 0}
    asyncio.create_task(
        run_upload_job(
            job_id=job_id,
            contents=contents,
            file_name=file.filename,
            skip_rows=skip_rows,
            mappings=mappings_obj,
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
            article_str = normalize_article(article)
            if not article_str:
                skipped += 1
                continue
            batch_articles.append(article_str)
            batch_payloads[article_str] = stock_value

        if batch_articles:
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
                article_value = normalize_article(point.payload.get("Артикул"))
                found_articles.add(article_value)
                qdrant_client.set_payload(
                    collection_name=collection_name,
                    payload={"Остаток": batch_payloads.get(article_value)},
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
        stock_jobs[job_id].update({"status": "completed", "progress": 100, **result})

    return result


async def run_stock_job(**kwargs):
    job_id = kwargs["job_id"]
    try:
        stock_jobs[job_id].update({"status": "resetting", "progress": 0})
        reset_stock_payload(kwargs["collection_name"])
        await process_stock_upload(**kwargs)
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
    contents = await file.read()
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


def ocr_pdf_bytes(contents: bytes) -> str:
    """
    Try multiple strategies to extract text from uploaded bytes.
    Strategies (in order):
    - If bytes look like an image, run OCR on the image via Pillow+pytesseract.
    - Try pdf2image + pytesseract.
    - Try repairing PDF with pikepdf and then pdf2image.
    - Try PyMuPDF (fitz) to extract text or rasterize pages and OCR.
    Returns combined text or empty string on failure.
    """
    def is_pdf(b: bytes) -> bool:
        try:
            # quick header check
            return b[:4] == b"%PDF"
        except Exception:
            return False

    # helper to OCR a PIL image
    def ocr_image(img) -> str:
        try:
            return pytesseract.image_to_string(img, lang="rus+eng") or ""
        except Exception:
            return ""

    # If not PDF, try to open as image
    if not is_pdf(contents):
        try:
            from PIL import Image
            from io import BytesIO

            img = Image.open(BytesIO(contents))
            return ocr_image(img).strip()
        except Exception:
            logger.debug("Not an image or failed to OCR non-PDF input", exc_info=True)

    # Try pdf2image -> pytesseract
    try:
        pages = convert_from_bytes(contents)
        extracted = []
        for page in pages:
            text = ocr_image(page)
            if text:
                extracted.append(text.strip())
        result = "\n".join(extracted).strip()
        if result:
            return result
    except Exception:
        logger.debug("pdf2image -> pytesseract failed", exc_info=True)

    # Try repairing PDF with pikepdf
    try:
        import pikepdf
        from io import BytesIO

        repaired = BytesIO()
        try:
            with pikepdf.open(BytesIO(contents)) as pdf:
                pdf.save(repaired)
            repaired_bytes = repaired.getvalue()
            pages = convert_from_bytes(repaired_bytes)
            extracted = []
            for page in pages:
                text = ocr_image(page)
                if text:
                    extracted.append(text.strip())
            result = "\n".join(extracted).strip()
            if result:
                return result
        except Exception:
            logger.debug("pikepdf repair attempt failed", exc_info=True)
    except Exception:
        logger.debug("pikepdf not available or failed", exc_info=True)

    # Try PyMuPDF as a last resort
    try:
        import fitz
        from PIL import Image
        from io import BytesIO

        doc = fitz.open(stream=contents, filetype="pdf")
        texts = []
        for page in doc:
            try:
                txt = page.get_text()
                if txt and txt.strip():
                    texts.append(txt.strip())
                    continue
            except Exception:
                logger.debug("PyMuPDF get_text failed for page, will rasterize", exc_info=True)

            try:
                pix = page.get_pixmap()
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                t = ocr_image(img)
                if t:
                    texts.append(t.strip())
            except Exception:
                logger.debug("PyMuPDF rasterize+OCR failed for page", exc_info=True)

        result = "\n".join(texts).strip()
        if result:
            return result
    except Exception:
        logger.debug("PyMuPDF not available or failed", exc_info=True)

    # give up
    return ""


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
            {"status": "running", "progress": 0, "indexed_chunks": 0, "total_chunks": total_chunks}
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
                    vector={"text-dense": dense_vector, "text-sparse": qdrant_sparse_vector},
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
        passports_jobs[job_id].update({"status": "completed", "progress": 100, **result})
    return result


async def run_passports_job(**kwargs):
    job_id = kwargs["job_id"]
    try:
        await process_passports_upload(**kwargs)
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
        file_entries.append((file.filename, await file.read()))

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

    points = qdrant_client.query_points(
        collection_name=collection_name,
        prefetch=[
            models.Prefetch(query=sparse_vector, using="text-sparse", limit=limit),
            models.Prefetch(query=dense_vector, using="text-dense", limit=limit),
        ],
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        limit=limit,
        with_payload=True,
        with_vectors=True,
    ).points
    return {"results": points}


async def run_catalog_search(
    *,
    query: str,
    collection_name: str,
    only_in_stock: bool = False,
    mode: str = "hybrid",
    limit: int = 15,
    candidate_limit: int = 10,
    dense_weight: float = 0.8,
    sparse_weight: float = 0.2,
    include_breakdown: bool = False,
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

    if limit <= 0:
        raise HTTPException(status_code=400, detail="limit must be > 0")
    if candidate_limit <= 0:
        raise HTTPException(status_code=400, detail="candidate_limit must be > 0")
    if dense_weight < 0 or sparse_weight < 0:
        raise HTTPException(status_code=400, detail="dense_weight and sparse_weight must be >= 0")
    if mode == "hybrid" and dense_weight == 0 and sparse_weight == 0:
        raise HTTPException(status_code=400, detail="For hybrid mode at least one of dense_weight/sparse_weight must be > 0")

    candidate_limit = max(candidate_limit, limit)

    hybrid_debug_last: Dict[str, Any] = {}

    async def query_points(limit: int, query_filter: Optional[models.Filter]):
        nonlocal hybrid_debug_last
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

        dense_points = qdrant_client.query_points(
            collection_name=collection_name,
            query=dense_vector,
            using="text-dense",
            limit=candidate_limit,
            with_payload=True,
            query_filter=query_filter,
        ).points
        sparse_points = qdrant_client.query_points(
            collection_name=collection_name,
            query=sparse_vector,
            using="text-sparse",
            limit=candidate_limit,
            with_payload=True,
            query_filter=query_filter,
        ).points

        dense_rank = {point.id: idx + 1 for idx, point in enumerate(dense_points)}
        sparse_rank = {point.id: idx + 1 for idx, point in enumerate(sparse_points)}

        rank_constant = 60.0

        def rrf(rank_value: Optional[int]) -> float:
            if not rank_value:
                return 0.0
            return 1.0 / (rank_constant + float(rank_value))

        candidates: Dict[Any, Any] = {}
        for point in dense_points:
            candidates[point.id] = point
        for point in sparse_points:
            if point.id not in candidates:
                candidates[point.id] = point

        weighted_scores: Dict[Any, float] = {}
        for point_id in candidates.keys():
            dense_rrf = rrf(dense_rank.get(point_id))
            sparse_rrf = rrf(sparse_rank.get(point_id))
            weighted_scores[point_id] = (dense_weight * dense_rrf) + (sparse_weight * sparse_rrf)

        ranked_ids = sorted(weighted_scores.keys(), key=lambda pid: weighted_scores[pid], reverse=True)
        final_points = [candidates[point_id] for point_id in ranked_ids[:limit]]
        for point in final_points:
            point.score = weighted_scores.get(point.id, 0.0)

        trace_limit = min(10, len(final_points))

        def compact_title(payload: Dict[str, Any]) -> str:
            article = str(payload.get("Артикул") or "").strip()
            name = str(payload.get("Наименование") or "").strip()
            if article and name:
                return f"{article} — {name}"
            return article or name or "n/a"

        breakdown = None
        if include_breakdown:
            breakdown_limit = min(5, candidate_limit)
            breakdown = {
                "dense_top": [
                    {
                        "id": str(point.id),
                        "rank": dense_rank.get(point.id),
                        "score": float(point.score or 0.0),
                        "title": compact_title(point.payload or {}),
                    }
                    for point in dense_points[:breakdown_limit]
                ],
                "sparse_top": [
                    {
                        "id": str(point.id),
                        "rank": sparse_rank.get(point.id),
                        "score": float(point.score or 0.0),
                        "title": compact_title(point.payload or {}),
                    }
                    for point in sparse_points[:breakdown_limit]
                ],
                "hybrid_top": [
                    {
                        "id": str(point.id),
                        "rank": idx + 1,
                        "score": weighted_scores.get(point.id),
                        "title": compact_title(point.payload or {}),
                    }
                    for idx, point in enumerate(final_points[:breakdown_limit])
                ],
            }

        hybrid_debug_last = {
            "ranking_stage": "weighted_rrf",
            "final_rank_source": "weighted_rrf",
            "rank_constant": rank_constant,
            "breakdown": breakdown,
            "trace_top": [
                {
                    "id": str(point.id),
                    "dense_rank": dense_rank.get(point.id),
                    "sparse_rank": sparse_rank.get(point.id),
                    "dense_rrf": rrf(dense_rank.get(point.id)),
                    "sparse_rrf": rrf(sparse_rank.get(point.id)),
                    "dense_weight": dense_weight,
                    "sparse_weight": sparse_weight,
                    "final_score": weighted_scores.get(point.id),
                }
                for point in final_points[:trace_limit]
            ]
        }

        return final_points

    in_stock_filter = models.Filter(
        must=[models.FieldCondition(key="Остаток", range=models.Range(gt=0))]
    )

    if only_in_stock:
        points = await query_points(limit, in_stock_filter)
        return {
            "results": points,
            "debug": {
                "mode": mode,
                "dense_dim": len(dense_vector) if dense_vector is not None else None,
                "sparse_nonzero": len(sparse_vector.indices) if sparse_vector is not None else None,
                "limit": limit,
                "candidate_limit": candidate_limit if mode == "hybrid" else None,
                "dense_weight": dense_weight if mode == "hybrid" else None,
                "sparse_weight": sparse_weight if mode == "hybrid" else None,
                "weights_applied": mode == "hybrid",
                "score_trace": hybrid_debug_last if mode == "hybrid" else None,
            },
        }

    in_stock_points = await query_points(min(5, limit), in_stock_filter)
    general_points = await query_points(limit, None)

    seen_ids = {point.id for point in in_stock_points}
    combined = list(in_stock_points)
    for point in general_points:
        if point.id in seen_ids:
            continue
        combined.append(point)
        seen_ids.add(point.id)
        if len(combined) >= limit:
            break

    return {
        "results": combined,
        "debug": {
            "mode": mode,
            "dense_dim": len(dense_vector) if dense_vector is not None else None,
            "sparse_nonzero": len(sparse_vector.indices) if sparse_vector is not None else None,
            "limit": limit,
            "candidate_limit": candidate_limit if mode == "hybrid" else None,
            "dense_weight": dense_weight if mode == "hybrid" else None,
            "sparse_weight": sparse_weight if mode == "hybrid" else None,
            "weights_applied": mode == "hybrid",
            "score_trace": hybrid_debug_last if mode == "hybrid" else None,
        },
    }


@app.get("/search")
async def search(
    query: str = Query(...),
    collection_name: str = Query(...),
    only_in_stock: bool = Query(False),
    mode: str = Query("hybrid"),
    limit: int = Query(15, ge=1, le=100),
    candidate_limit: int = Query(10, ge=1, le=200),
    dense_weight: float = Query(0.8, ge=0),
    sparse_weight: float = Query(0.2, ge=0),
    include_breakdown: bool = Query(False),
):
    try:
        return await run_catalog_search(
            query=query,
            collection_name=collection_name,
            only_in_stock=only_in_stock,
            mode=mode,
            limit=limit,
            candidate_limit=candidate_limit,
            dense_weight=dense_weight,
            sparse_weight=sparse_weight,
            include_breakdown=include_breakdown,
        )
    except HTTPException:
        raise
    except Exception as e:
        message = str(e)
        low = message.lower()
        if "doesn't exist" in low or ("collection" in low and "not found" in low):
            raise HTTPException(
                status_code=404,
                detail=(
                    f"Коллекция '{collection_name}' не найдена. "
                    "Укажите существующую коллекцию или загрузите прайс в эту коллекцию."
                ),
            )
        logger.error("Search failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=message)


@app.get("/collections")
async def collections():
    try:
        result = qdrant_client.get_collections()
        names = [item.name for item in result.collections]
        return {"collections": names}
    except Exception as exc:
        logger.error("Failed to list collections: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))


@app.delete("/collection")
async def delete_collection(collection_name: str = Query(...)):
    try:
        qdrant_client.delete_collection(collection_name=collection_name)
        return {"status": "deleted", "collection_name": collection_name}
    except Exception as exc:
        logger.error("Failed to delete collection %s: %s", collection_name, exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))
