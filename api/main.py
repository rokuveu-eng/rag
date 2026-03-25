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
import subprocess
import tempfile
from io import BytesIO


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

# Optional default Bitrix webhook URL (incoming webhook)
BITRIX_WEBHOOK_URL = os.getenv("BITRIX_WEBHOOK_URL", "").strip()

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


def extract_docx_text(contents: bytes) -> str:
    try:
        from docx import Document
        from PIL import Image
        import zipfile
        import xml.etree.ElementTree as ET

        # First try python-docx (works for most .docx files)
        parts = []
        try:
            doc = Document(BytesIO(contents))
            for p in doc.paragraphs:
                t = p.text.strip()
                if t:
                    parts.append(t)

            # attempt to extract images referenced by relationships and OCR them
            ocr_texts = []
            try:
                for rel in doc.part.rels.values():
                    try:
                        target = getattr(rel, "target_part", None)
                        content_type = getattr(target, "content_type", "")
                        if target is not None and content_type.startswith("image"):
                            blob = target.blob
                            try:
                                img = Image.open(BytesIO(blob))
                                txt = pytesseract.image_to_string(img, lang="rus+eng") or ""
                                if txt.strip():
                                    ocr_texts.append(txt.strip())
                            except Exception:
                                logger.debug("Failed OCR image in docx (doc.part)", exc_info=True)
                    except Exception:
                        continue
            except Exception:
                logger.debug("No images extracted from docx via doc.part", exc_info=True)

            if ocr_texts:
                parts.append("\n".join(ocr_texts))

        except Exception:
            logger.debug("python-docx path failed, will try zip/xml fallback", exc_info=True)

        # If python-docx didn't yield text, try unzip and parse document.xml
        if not parts:
            try:
                with zipfile.ZipFile(BytesIO(contents)) as z:
                    # extract document.xml text nodes
                    if "word/document.xml" in z.namelist():
                        xml_bytes = z.read("word/document.xml")
                        # parse XML and extract text from w:t elements
                        ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
                        root = ET.fromstring(xml_bytes)
                        texts = [t.text for t in root.findall('.//w:t', ns) if t.text]
                        if texts:
                            parts.extend([t.strip() for t in texts if t.strip()])

                    # OCR images in word/media/
                    media_files = [name for name in z.namelist() if name.startswith("word/media/")]
                    ocr_texts = []
                    for m in media_files:
                        try:
                            blob = z.read(m)
                            img = Image.open(BytesIO(blob))
                            txt = pytesseract.image_to_string(img, lang="rus+eng") or ""
                            if txt.strip():
                                ocr_texts.append(txt.strip())
                        except Exception:
                            logger.debug("Failed OCR media image %s", m, exc_info=True)
                    if ocr_texts:
                        parts.append("\n".join(ocr_texts))
            except Exception:
                logger.debug("zip/xml fallback for docx failed", exc_info=True)

        return "\n".join(parts).strip()
    except Exception:
        logger.debug("extract_docx_text failed (final)", exc_info=True)
        return ""


def extract_doc_text(contents: bytes) -> str:
    # try antiword for legacy .doc
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.doc') as tf:
            tf.write(contents)
            tf.flush()
            tmp_path = tf.name
        try:
            res = subprocess.run(["antiword", tmp_path], capture_output=True, text=True, timeout=30)
            if res.returncode == 0:
                return res.stdout.strip()
        except FileNotFoundError:
            logger.debug("antiword not installed")
        except Exception:
            logger.debug("antiword failed", exc_info=True)
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass
    return ""


async def bitrix_call(base_webhook: str, method: str, params: dict):
    """Call Bitrix incoming webhook REST method.
    `base_webhook` should be the incoming webhook base URL without trailing slash,
    e.g. https://your.bitrix24.ru/rest/1/XXXXX
    We POST to {base_webhook}/{method} with params as form data.
    """
    url = base_webhook.rstrip("/") + "/" + method
    async with httpx.AsyncClient(timeout=60.0) as client:
        try:
            resp = await client.post(url, data=params)
            resp.raise_for_status()
            return resp.json()
        except Exception:
            logger.exception("Bitrix API call failed: %s %s", url, params)
            return None


async def list_bitrix_folder_recursive(base_webhook: str, folder_id: str, file_types: Optional[List[str]] = None):
    """Return list of file descriptors: dicts with keys `name`, `id`, `download_url` (may be None).
    Recurses into subfolders.
    """
    results = []

    async def walk(fid: str):
        resp = await bitrix_call(base_webhook, "disk.folder.getchildren", {"id": fid})
        if not resp or "result" not in resp:
            return
        items = resp.get("result", [])
        for it in items:
            try:
                it_type = it.get("TYPE") or it.get("type") or it.get("TYPE_ID")
                if str(it_type).lower() == "folder" or it.get("TYPE") == "folder":
                    sub_id = it.get("ID") or it.get("id")
                    if sub_id:
                        await walk(sub_id)
                    continue
                # treat as file
                file_id = it.get("ID") or it.get("id") or it.get("FILE_ID")
                name = it.get("NAME") or it.get("name") or it.get("TITLE") or it.get("ORIGINAL_NAME")
                if not file_id:
                    continue
                # fetch file info to get download URL
                finfo = await bitrix_call(base_webhook, "disk.file.get", {"id": file_id})
                download_url = None
                if finfo and "result" in finfo:
                    f = finfo.get("result")
                    # common field
                    download_url = f.get("DOWNLOAD_URL") or f.get("downloadUrl")
                    # some responses nest file data
                    if not download_url and isinstance(f, dict):
                        for k in ("file", "FILE", "fileInfo"):
                            if k in f and isinstance(f[k], dict):
                                download_url = f[k].get("DOWNLOAD_URL") or f[k].get("downloadUrl")
                                if download_url:
                                    break
                # filter by extension if requested
                if file_types:
                    lower = (name or "").lower()
                    ok = any(lower.endswith(ext) for ext in file_types)
                    if not ok:
                        continue

                results.append({"name": name or f"file_{file_id}", "id": file_id, "download_url": download_url})
            except Exception:
                logger.exception("Error processing item from bitrix folder listing: %s", it)

    await walk(folder_id)
    return results


async def download_from_url(url: str) -> Optional[bytes]:
    if not url:
        return None
    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            r = await client.get(url)
            r.raise_for_status()
            return r.content
    except Exception:
        logger.exception("Failed to download file from %s", url)
        return None


async def process_bitrix_import(*, base_webhook: str, folder_id: str, collection_name: str, file_types: Optional[List[str]], batch_size: int, points_batch_size: int, job_id: str):
    # recreate collection (delete + create)
    try:
        qdrant_client.recreate_collection(
            collection_name=collection_name,
            vectors_config={"text-dense": models.VectorParams(size=1024, distance=models.Distance.COSINE)},
            sparse_vectors_config={"text-sparse": models.SparseVectorParams(index=models.SparseIndexParams(on_disk=False))},
        )
    except Exception:
        logger.exception("Failed to recreate collection %s", collection_name)

    if job_id:
        passports_jobs[job_id].update({"status": "running", "progress": 0})

    files = await list_bitrix_folder_recursive(base_webhook, folder_id, file_types)
    file_entries = []
    total = len(files)
    counted = 0
    for f in files:
        name = f.get("name")
        download_url = f.get("download_url")
        contents = None
        if download_url:
            contents = await download_from_url(download_url)
        else:
            # try disk.file.getContent fallback
            finfo = await bitrix_call(base_webhook, "disk.file.getContent", {"id": f.get("id")})
            # disk.file.getContent may return redirect URL or raw content; skip if not usable
            contents = None

        if not contents:
            logger.warning("Skipped Bitrix file %s (no content downloaded)", name)
            continue
        file_entries.append((name, contents, download_url))
        counted += 1
        if job_id:
            passports_jobs[job_id].update({"indexed_chunks": counted, "total_chunks": total, "progress": round((counted / total) * 100 if total else 0, 2)})

    # hand off to existing passports processing (it will attach download_url from tuples)
    if file_entries:
        await process_passports_upload(file_entries=file_entries, collection_name=collection_name, batch_size=batch_size, points_batch_size=points_batch_size, job_id=job_id)
    else:
        raise HTTPException(status_code=400, detail="No files found or downloaded from Bitrix folder")


async def run_bitrix_import_job(**kwargs):
    job_id = kwargs.get("job_id")
    try:
        await process_bitrix_import(**kwargs)
    except Exception as exc:
        logger.error("Bitrix import job failed: %s", exc, exc_info=True)
        passports_jobs[job_id].update({"status": "failed", "error": str(exc)})


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


def build_passport_documents(file_entries: List[tuple], job_id: Optional[str] = None):
    """
    Build chunk documents and collect per-document metadata.
    Returns: (documents, payloads, skipped, docs_meta)
    docs_meta: {doc_id: {file_name, download_url, full_text, summary, category}}
    """
    documents: List[str] = []
    payloads: List[Dict[str, Any]] = []
    skipped = 0
    docs_meta: Dict[str, Dict[str, Any]] = {}

    def categorize_document(text_snippet: str) -> str:
        txt = (text_snippet or '').lower()
        mapping = {
            'logistics': ['доставка', 'отгрузк', 'логист', 'терминал', 'самовывоз'],
            'payment': ['оплат', 'счет', 'безнал', 'наличн'],
            'warranty': ['гарант', 'ремонт', 'сервис'],
            'safety': ['техника безопасности', 'охрана труда', 'требован'],
        }
        for cat, keywords in mapping.items():
            for kw in keywords:
                if kw in txt:
                    return cat
        return 'general'

    for entry in file_entries:
        if not (isinstance(entry, (tuple, list)) and len(entry) >= 2):
            logger.warning("Skipped invalid file entry: %s", entry)
            skipped += 1
            continue

        filename, contents = entry[0], entry[1]
        download_url = entry[2] if len(entry) >= 3 else None

        lower = (filename or '').lower()
        text = ''
        src = 'auto'
        if lower.endswith('.pdf'):
            text = ocr_pdf_bytes(contents)
            src = 'pdf'
        elif lower.endswith('.docx'):
            text = extract_docx_text(contents)
            src = 'docx'
        elif lower.endswith('.doc'):
            text = extract_doc_text(contents)
            src = 'doc'
        elif lower.endswith('.txt'):
            try:
                text = contents.decode('utf-8', errors='ignore')
                src = 'txt'
            except Exception:
                text = ''
        else:
            text = ocr_pdf_bytes(contents)
            src = 'auto'

        if not text:
            try:
                head = contents[:512]
                logger.warning("Skipped file %s — no text extracted; head=%s", filename, head[:128])
            except Exception:
                logger.warning("Skipped file %s — no text extracted", filename)

            try:
                failed_dir = os.path.join('failed_uploads')
                os.makedirs(failed_dir, exist_ok=True)
                safe_name = f"{job_id or 'noj'}_{uuid.uuid4().hex}_{os.path.basename(filename)}"
                failed_path = os.path.join(failed_dir, safe_name)
                with open(failed_path, 'wb') as wf:
                    wf.write(contents)
                logger.info("Saved skipped file to %s", failed_path)
            except Exception:
                logger.exception("Failed to save skipped file %s", filename)

            skipped += 1
            continue

        # deterministic doc id from download_url or filename
        base_for_id = download_url or filename or str(uuid.uuid4())
        doc_id = str(uuid.uuid5(uuid.NAMESPACE_URL, base_for_id))

        # aggregate full text per document
        if doc_id not in docs_meta:
            docs_meta[doc_id] = {"file_name": filename, "download_url": download_url, "full_text": text}
        else:
            docs_meta[doc_id]["full_text"] += '\n' + text

        # create summary (extractive)
        try:
            sentences = re.split(r'(?<=[.!?])\s+', text.strip())
            if len(sentences) <= 3:
                summary = ' '.join(sentences)[:1200]
            else:
                summary = ' '.join(sentences[:3])[:1200]
        except Exception:
            summary = (text or '')[:1200]

        docs_meta[doc_id]['summary'] = summary
        docs_meta[doc_id]['category'] = categorize_document(summary)

        chunks = chunk_text(text)
        if not chunks:
            skipped += 1
            continue

        for chunk_index, chunk in enumerate(chunks, start=1):
            documents.append(chunk)
            p = {
                'file_name': filename,
                'source': src,
                'text': chunk,
                'chunk_id': chunk_index,
                'chunks_total': len(chunks),
                'doc_id': doc_id,
                'category': docs_meta[doc_id]['category'],
                'is_doc': False,
            }
            if download_url:
                p['download_url'] = download_url
            payloads.append(p)

    return documents, payloads, skipped, docs_meta


async def process_passports_upload(
    *,
    file_entries: List[tuple],
    collection_name: str,
    batch_size: int = 8,
    points_batch_size: int = 200,
    job_id: Optional[str] = None,
):
    """
    Ingest chunks and also add document-level points (summaries + category).
    """
    create_collection(collection_name)

    documents, payloads, skipped, docs_meta = build_passport_documents(file_entries, job_id=job_id)

    total_chunks = len(documents)
    indexed = 0
    total_start = perf_counter()

    if job_id:
        passports_jobs[job_id].update({
            'status': 'running',
            'progress': 0,
            'indexed_chunks': 0,
            'total_chunks': total_chunks,
        })

    # First: create and upsert document-level embeddings (one per doc)
    # Automatic category classification using embeddings (override simple keyword heuristics)
    try:
        categories = ['general', 'logistics', 'payment', 'warranty', 'safety']
        # compute embeddings for summaries and category labels together
        if docs_meta:
            doc_ids = list(docs_meta.keys())
            summaries = [docs_meta[d]['summary'] for d in doc_ids]
            combined = summaries + categories
            emb_all = await get_ollama_embeddings(combined)
            doc_embs = emb_all[: len(summaries)]
            cat_embs = emb_all[len(summaries) :]
            for i, did in enumerate(doc_ids):
                best_label = 'general'
                best_score = -1.0
                for c_name, c_emb in zip(categories, cat_embs):
                    score = cosine_similarity(doc_embs[i], c_emb)
                    if score > best_score:
                        best_score = score
                        best_label = c_name
                docs_meta[did]['category'] = best_label
    except Exception:
        logger.exception("Failed to classify document categories via embeddings")

    try:
        doc_ids = list(docs_meta.keys())
        summaries = [docs_meta[d]['summary'] for d in doc_ids]
        if summaries:
            doc_embeddings = await get_ollama_embeddings(summaries)
            doc_points = []
            for did, emb in zip(doc_ids, doc_embeddings):
                meta = docs_meta[did]
                payload = {
                    'is_doc': True,
                    'doc_id': did,
                    'file_name': meta.get('file_name'),
                    'download_url': meta.get('download_url'),
                    'doc_summary': meta.get('summary'),
                    'category': meta.get('category'),
                }
                doc_points.append(
                    models.PointStruct(id=f"doc-{did}", vector={"text-dense": emb}, payload=payload)
                )
            if doc_points:
                safe_upsert(collection_name, doc_points)
    except Exception:
        logger.exception("Failed to create document-level points")

    # Now ingest chunk-level points in batches
    for start in range(0, total_chunks, batch_size):
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
            payload = batch_payloads[offset]
            point_id = str(uuid.uuid4())
            batch_points.append(
                models.PointStruct(
                    id=point_id,
                    vector={"text-dense": dense_vector, "text-sparse": qdrant_sparse_vector},
                    payload=payload,
                )
            )

        for chunk_start in range(0, len(batch_points), points_batch_size):
            chunk = batch_points[chunk_start : chunk_start + points_batch_size]
            safe_upsert(collection_name, chunk)
            indexed += len(chunk)

            if job_id and total_chunks:
                percent = (indexed / total_chunks) * 100 if total_chunks else 100
                passports_jobs[job_id].update(
                    {
                        'status': 'running',
                        'progress': round(percent, 2),
                        'indexed_chunks': indexed,
                        'total_chunks': total_chunks,
                    }
                )

    duration = perf_counter() - total_start
    result = {
        'status': 'success',
        'indexed_chunks': indexed,
        'skipped_files': skipped,
        'total_chunks': total_chunks,
        'duration_sec': round(duration, 3),
    }
    if job_id:
        passports_jobs[job_id].update({'status': 'completed', 'progress': 100, **result})
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


@app.post("/import_bitrix_folder")
async def import_bitrix_folder(
    webhook_url: Optional[str] = Form(None),
    folder_id: str = Form(...),
    collection_name: str = Form(...),
    file_types: Optional[str] = Form(None, description="Comma-separated list of extensions, e.g. .pdf,.docx"),
    batch_size: int = Form(8),
    points_batch_size: int = Form(200),
):
    """Import files recursively from a Bitrix24 folder (uses incoming webhook REST URL).

    Provide either `webhook_url` or set `BITRIX_WEBHOOK_URL` in env.
    The endpoint will recreate the `collection_name` and reindex files.
    """
    base = (webhook_url or BITRIX_WEBHOOK_URL or "").strip()
    if not base:
        raise HTTPException(status_code=400, detail="No Bitrix webhook URL provided; set BITRIX_WEBHOOK_URL or pass webhook_url")

    types = None
    if file_types:
        types = [t.strip().lower() for t in file_types.split(",") if t.strip()]

    job_id = str(uuid.uuid4())
    passports_jobs[job_id] = {"status": "queued", "progress": 0}
    asyncio.create_task(
        run_bitrix_import_job(
            job_id=job_id,
            base_webhook=base,
            folder_id=folder_id,
            collection_name=collection_name,
            file_types=types,
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
    limit: int = Query(5, ge=1, le=50),
    only_payload: bool = Query(False),
    by_document: bool = Query(False, description="If true, perform doc->passage two-stage retrieval"),
    category: Optional[str] = Query(None, description="Optional category filter to restrict search"),
    doc_top_k: int = Query(5, ge=1, le=50, description="How many top documents to retrieve in stage 1"),
    chunks_per_doc: int = Query(3, ge=1, le=10, description="How many chunks to fetch per top doc"),
):
    dense_vector = await get_ollama_embedding(query)
    sparse_vector_gen = list(sparse_embedding_model.embed([query]))[0]
    sparse_vector = models.SparseVector(
        indices=sparse_vector_gen.indices.tolist(),
        values=sparse_vector_gen.values.tolist(),
    )

    # build optional category filter
    query_filter = None
    if category:
        query_filter = models.Filter(must=[models.FieldCondition(key="category", match=models.MatchValue(value=category))])

    if not by_document:
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
            query_filter=query_filter,
        ).points
        if only_payload:
            return {"results": [getattr(p, "payload", {}) for p in points]}
        return {"results": points}

    # by_document two-stage retrieval
    # Stage 1: retrieve top documents (points where is_doc == True)
    doc_filter_components = [models.FieldCondition(key="is_doc", match=models.MatchValue(value=True))]
    if category:
        doc_filter_components.append(models.FieldCondition(key="category", match=models.MatchValue(value=category)))
    doc_filter = models.Filter(must=doc_filter_components)

    dense_docs = qdrant_client.query_points(
        collection_name=collection_name,
        query=dense_vector,
        using="text-dense",
        limit=doc_top_k,
        with_payload=True,
        query_filter=doc_filter,
    ).points
    sparse_docs = qdrant_client.query_points(
        collection_name=collection_name,
        query=sparse_vector,
        using="text-sparse",
        limit=doc_top_k,
        with_payload=True,
        query_filter=doc_filter,
    ).points

    # fuse doc scores via RRF-like weighting
    dense_rank = {p.id: i + 1 for i, p in enumerate(dense_docs)}
    sparse_rank = {p.id: i + 1 for i, p in enumerate(sparse_docs)}
    rank_constant = 60.0

    def rrf(rank_value: Optional[int]) -> float:
        if not rank_value:
            return 0.0
        return 1.0 / (rank_constant + float(rank_value))

    doc_candidates = {p.id: p for p in dense_docs}
    for p in sparse_docs:
        if p.id not in doc_candidates:
            doc_candidates[p.id] = p

    weighted_scores = {}
    for pid in doc_candidates:
        weighted_scores[pid] = (rrf(dense_rank.get(pid)) * 0.8) + (rrf(sparse_rank.get(pid)) * 0.2)

    ranked_doc_ids = sorted(weighted_scores.keys(), key=lambda k: weighted_scores[k], reverse=True)[:doc_top_k]
    top_docs = [doc_candidates[did] for did in ranked_doc_ids]

    # Stage 2: for each top doc, fetch top chunks restricted to doc_id
    final_chunks = []
    for doc_point in top_docs:
        doc_payload = getattr(doc_point, 'payload', {}) or {}
        did = doc_payload.get('doc_id') or (str(doc_point.id).replace('doc-', ''))
        chunk_filter = models.Filter(must=[models.FieldCondition(key='doc_id', match=models.MatchValue(value=did))])
        if category:
            chunk_filter.must.append(models.FieldCondition(key='category', match=models.MatchValue(value=category)))

        chunks = qdrant_client.query_points(
            collection_name=collection_name,
            query=dense_vector,
            using='text-dense',
            limit=chunks_per_doc,
            with_payload=True,
            with_vectors=True,
            query_filter=chunk_filter,
        ).points
        # attach parent doc metadata
        for c in chunks:
            if not c.payload:
                c.payload = {}
            c.payload['_parent_doc'] = {
                'doc_id': did,
                'file_name': doc_payload.get('file_name'),
                'doc_summary': doc_payload.get('doc_summary'),
                'category': doc_payload.get('category'),
            }
            final_chunks.append(c)

    # limit to requested number
    results = final_chunks[:limit]
    if only_payload:
        return {"results": [getattr(p, 'payload', {}) for p in results]}
    return {"results": results}


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
