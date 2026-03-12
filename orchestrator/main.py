#!/usr/bin/env python
# -*- coding: utf-8 -*-
import json
import os
import re
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from openpyxl import Workbook
from pydantic import BaseModel, Field
from redis import asyncio as redis_async


CATALOG_API_URL = os.getenv("CATALOG_API_URL", "http://api:8424").rstrip("/")
DEFAULT_AI_BASE_URL = os.getenv("AI_BASE_URL", "https://polza.ai/api/v1").rstrip("/")
DEFAULT_AI_MODEL = os.getenv("AI_MODEL", "openai/gpt-4o")
SPEC_DIR = Path(os.getenv("SPEC_DIR", "/app/specs"))
REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379/0")
MEMORY_TTL_SEC = int(os.getenv("MEMORY_TTL_SEC", "86400"))
MEMORY_MAX_TURNS = int(os.getenv("MEMORY_MAX_TURNS", "8"))
MEMORY_PREFIX = os.getenv("MEMORY_PREFIX", "mem")
SPEC_DIR.mkdir(parents=True, exist_ok=True)


app = FastAPI(title="AI Orchestrator")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

redis_client: Optional[redis_async.Redis] = None


class AgentRequest(BaseModel):
    message: str = Field(..., min_length=1)
    collection_name: str = Field(..., min_length=1)
    ai_base_url: Optional[str] = None
    ai_api_key: Optional[str] = None
    ai_model: Optional[str] = None
    probable_limit: int = Field(3, ge=1, le=10)
    tenant_id: str = Field("default", min_length=1)
    dialog_id: Optional[str] = None
    reset_memory: bool = False


class MemoryResetRequest(BaseModel):
    tenant_id: str = Field(..., min_length=1)
    dialog_id: str = Field(..., min_length=1)


class ParsedItem(BaseModel):
    query_text: str
    qty: float = 1
    unit: Optional[str] = None


@dataclass
class RowResult:
    query_text: str
    replacement_article: str
    replacement_name: str
    score: Optional[float]
    score_norm: Optional[float]
    price: Optional[float]
    qty: float
    total: Optional[float]
    probable: List[str]
    dense_top: List[str]
    sparse_top: List[str]
    hybrid_top: List[str]


def parse_number(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip().replace(" ", "")
    if not text:
        return None
    text = text.replace(",", ".")
    match = re.search(r"-?\d+(?:\.\d+)?", text)
    if not match:
        return None
    try:
        return float(match.group(0))
    except ValueError:
        return None


def extract_price(payload: Dict[str, Any]) -> Optional[float]:
    for key in ["Тариф с НДС, руб", "Цена", "price", "стоимость"]:
        if key in payload:
            price = parse_number(payload.get(key))
            if price is not None:
                return price
    return None


def payload_title(payload: Dict[str, Any]) -> str:
    article = payload.get("Артикул")
    name = payload.get("Наименование")
    if article and name:
        return f"{article} — {name}"
    if name:
        return str(name)
    if article:
        return str(article)
    return "Не найдено"


def fallback_parse_items(message: str) -> List[ParsedItem]:
    lines = [line.strip(" -•\t") for line in re.split(r"[\n;]+", message) if line.strip()]
    if not lines:
        return [ParsedItem(query_text=message.strip(), qty=1)]

    items: List[ParsedItem] = []
    for line in lines:
        qty = 1.0
        unit = "шт"
        m = re.search(r"(\d+(?:[\.,]\d+)?)\s*(шт|pcs|ед|м|кг)?", line, re.IGNORECASE)
        if m:
            qty = parse_number(m.group(1)) or 1.0
            if m.group(2):
                unit = m.group(2)
        query_text = re.sub(r"\b\d+(?:[\.,]\d+)?\s*(шт|pcs|ед|м|кг)?\b", "", line, flags=re.IGNORECASE).strip(" ,")
        items.append(ParsedItem(query_text=query_text or line, qty=qty, unit=unit))
    return items


def memory_key(tenant_id: str, dialog_id: str) -> str:
    return f"{MEMORY_PREFIX}:{tenant_id}:{dialog_id}"


async def get_redis_client() -> Optional[redis_async.Redis]:
    global redis_client
    if not REDIS_URL:
        return None
    if redis_client is None:
        redis_client = redis_async.from_url(REDIS_URL, decode_responses=True)
    return redis_client


async def load_short_memory(tenant_id: str, dialog_id: str) -> List[Dict[str, str]]:
    client = await get_redis_client()
    if not client:
        return []
    raw = await client.get(memory_key(tenant_id, dialog_id))
    if not raw:
        return []
    try:
        data = json.loads(raw)
        if isinstance(data, list):
            return [
                {
                    "role": str(item.get("role") or ""),
                    "content": str(item.get("content") or ""),
                }
                for item in data
                if isinstance(item, dict)
            ]
    except Exception:
        return []
    return []


async def append_short_memory(tenant_id: str, dialog_id: str, role: str, content: str) -> None:
    if not content.strip():
        return
    client = await get_redis_client()
    if not client:
        return
    history = await load_short_memory(tenant_id, dialog_id)
    history.append({"role": role, "content": content.strip()})
    history = history[-MEMORY_MAX_TURNS:]
    await client.setex(memory_key(tenant_id, dialog_id), MEMORY_TTL_SEC, json.dumps(history, ensure_ascii=False))


async def reset_short_memory(tenant_id: str, dialog_id: str) -> bool:
    client = await get_redis_client()
    if not client:
        return False
    deleted = await client.delete(memory_key(tenant_id, dialog_id))
    return bool(deleted)


async def parse_items_with_llm(
    req: AgentRequest,
    memory_context: Optional[List[Dict[str, str]]] = None,
) -> Tuple[List[ParsedItem], Dict[str, Any]]:
    ai_base_url = (req.ai_base_url or DEFAULT_AI_BASE_URL).rstrip("/")
    ai_model = req.ai_model or DEFAULT_AI_MODEL
    if not req.ai_api_key:
        return fallback_parse_items(req.message), {"parser": "fallback", "reason": "empty_api_key"}

    schema = {
        "name": "request_items",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "items": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "query_text": {"type": "string"},
                            "qty": {"type": "number"},
                            "unit": {"type": "string"},
                        },
                        "required": ["query_text", "qty", "unit"],
                        "additionalProperties": False,
                    },
                },
                "need_spec": {"type": "boolean"},
            },
            "required": ["items", "need_spec"],
            "additionalProperties": False,
        },
    }

    messages: List[Dict[str, str]] = [
        {
            "role": "system",
            "content": (
                "Ты парсер заявок по электротехническому прайс-листу. "
                "Верни только JSON по схеме. qty всегда число. "
                "Если количество не указано, qty=1 и unit='шт'."
            ),
        }
    ]
    for turn in (memory_context or [])[-MEMORY_MAX_TURNS:]:
        role = str(turn.get("role") or "").strip().lower()
        content = str(turn.get("content") or "").strip()
        if role in {"user", "assistant"} and content:
            messages.append({"role": role, "content": content})
    messages.append({"role": "user", "content": req.message})

    payload = {
        "model": ai_model,
        "messages": messages,
        "response_format": {"type": "json_schema", "json_schema": schema},
        "temperature": 0,
    }
    headers = {"Authorization": f"Bearer {req.ai_api_key}"}

    try:
        async with httpx.AsyncClient(timeout=40.0) as client:
            response = await client.post(f"{ai_base_url}/chat/completions", json=payload, headers=headers)
            response.raise_for_status()
            data = response.json()
        content = data.get("choices", [{}])[0].get("message", {}).get("content", "{}")
        parsed = json.loads(content)
        items = [ParsedItem(**item) for item in parsed.get("items", []) if item.get("query_text")]
        if not items:
            raise ValueError("LLM returned empty items")
        usage = data.get("usage", {})
        return items, {
            "parser": "llm",
            "usage": usage,
            "model": data.get("model"),
            "memory_turns_used": len((memory_context or [])[-MEMORY_MAX_TURNS:]),
        }
    except Exception:
        return fallback_parse_items(req.message), {"parser": "fallback", "reason": "llm_parse_failed"}


async def catalog_search(query: str, collection_name: str) -> Dict[str, Any]:
    params = {
        "query": query,
        "collection_name": collection_name,
        "mode": "hybrid",
        "only_in_stock": "false",
        "include_breakdown": "true",
    }
    async with httpx.AsyncClient(timeout=40.0) as client:
        response = await client.get(f"{CATALOG_API_URL}/search", params=params)
        if response.status_code >= 400:
            detail = ""
            try:
                payload = response.json()
                detail = str(payload.get("detail") or payload)
            except Exception:
                detail = response.text or ""

            low = detail.lower()
            if "doesn't exist" in low or "collection" in low and "not found" in low:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"Коллекция '{collection_name}' не найдена. "
                        "Загрузите прайс в эту коллекцию или укажите существующую коллекцию в AI чате."
                    ),
                )

            raise HTTPException(
                status_code=502,
                detail=f"Ошибка Catalog API ({response.status_code}): {detail or 'unknown error'}",
            )
        return response.json()


def map_row(
    item: ParsedItem,
    points: List[Dict[str, Any]],
    probable_limit: int,
    breakdown: Optional[Dict[str, Any]] = None,
) -> RowResult:
    qty = item.qty if item.qty and item.qty > 0 else 1.0
    if not points:
        return RowResult(
            query_text=item.query_text,
            replacement_article="n/a",
            replacement_name="n/a",
            score=None,
            score_norm=None,
            price=None,
            qty=qty,
            total=None,
            probable=[],
            dense_top=[],
            sparse_top=[],
            hybrid_top=[],
        )

    best = points[0]
    best_payload = best.get("payload") or {}
    replacement_article = str(best_payload.get("Артикул") or "n/a")
    replacement_name = str(best_payload.get("Наименование") or "n/a")
    score = parse_number(best.get("score"))

    all_scores = [parse_number(p.get("score")) for p in points]
    valid_scores = [s for s in all_scores if s is not None]
    score_norm: Optional[float] = None
    if score is not None and valid_scores:
        top_score = max(valid_scores)
        low_score = min(valid_scores)
        if top_score > low_score:
            score_norm = round((score - low_score) / (top_score - low_score), 4)
        else:
            score_norm = 1.0

    price = extract_price(best_payload)
    total = round(price * qty, 2) if price is not None else None

    probable: List[str] = []
    for point in points[1 : 1 + probable_limit]:
        payload = point.get("payload") or {}
        title = payload_title(payload)
        candidate_price = extract_price(payload)
        if candidate_price is not None:
            probable.append(f"{title} (цена: {candidate_price})")
        else:
            probable.append(title)

    dense_top = [str(item.get("title") or "n/a") for item in (breakdown or {}).get("dense_top", [])]
    sparse_top = [str(item.get("title") or "n/a") for item in (breakdown or {}).get("sparse_top", [])]
    hybrid_top = [str(item.get("title") or "n/a") for item in (breakdown or {}).get("hybrid_top", [])]

    return RowResult(
        query_text=item.query_text,
        replacement_article=replacement_article,
        replacement_name=replacement_name,
        score=score,
        score_norm=score_norm,
        price=price,
        qty=qty,
        total=total,
        probable=probable,
        dense_top=dense_top,
        sparse_top=sparse_top,
        hybrid_top=hybrid_top,
    )


def build_reply(rows: List[RowResult], parser_meta: Dict[str, Any], spec_file_id: Optional[str] = None) -> str:
    lines = ["Результат подбора:"]
    for index, row in enumerate(rows, start=1):
        lines.append(f"{index}. Запрос: {row.query_text}")
        lines.append(f"   Артикул замены: {row.replacement_article}")
        lines.append(f"   Наименование замены: {row.replacement_name}")
        lines.append(f"   Score: {row.score if row.score is not None else 'n/a'}")
        lines.append(f"   ScoreNorm(0..1): {row.score_norm if row.score_norm is not None else 'n/a'}")
        lines.append(f"   Цена: {row.price if row.price is not None else 'n/a'}")
        lines.append(f"   Кол-во: {row.qty}")
        lines.append(f"   Сумма: {row.total if row.total is not None else 'n/a'}")
        lines.append(
            "   Вероятные замены: " + ("; ".join(row.probable) if row.probable else "нет")
        )
        lines.append("   Dense top: " + ("; ".join(row.dense_top) if row.dense_top else "нет"))
        lines.append("   Sparse top: " + ("; ".join(row.sparse_top) if row.sparse_top else "нет"))
        lines.append("   Hybrid top: " + ("; ".join(row.hybrid_top) if row.hybrid_top else "нет"))
    if spec_file_id:
        lines.append(f"Файл спецификации: /agent/files/{spec_file_id}")
    lines.append(f"Parser: {parser_meta.get('parser')}")
    return "\n".join(lines)


def write_spec_xlsx(rows: List[RowResult]) -> str:
    file_id = f"spec_{uuid.uuid4().hex}.xlsx"
    file_path = SPEC_DIR / file_id

    wb = Workbook()
    ws = wb.active
    ws.title = "Спецификация"
    ws.append(
        [
            "Запрос",
            "Артикул замены",
            "Наименование замены",
            "Score",
            "ScoreNorm(0..1)",
            "Цена",
            "Кол-во",
            "Сумма",
            "Вероятные замены",
            "Dense top",
            "Sparse top",
            "Hybrid top",
        ]
    )

    for row in rows:
        ws.append(
            [
                row.query_text,
                row.replacement_article,
                row.replacement_name,
                row.score,
                row.score_norm,
                row.price,
                row.qty,
                row.total,
                "; ".join(row.probable),
                "; ".join(row.dense_top),
                "; ".join(row.sparse_top),
                "; ".join(row.hybrid_top),
            ]
        )

    wb.save(file_path)
    return file_id


async def run_agent(req: AgentRequest, force_spec: bool = False) -> Dict[str, Any]:
    if req.reset_memory and req.dialog_id:
        await reset_short_memory(req.tenant_id, req.dialog_id)

    memory_context: List[Dict[str, str]] = []
    if req.dialog_id:
        memory_context = await load_short_memory(req.tenant_id, req.dialog_id)

    items, parser_meta = await parse_items_with_llm(req, memory_context=memory_context)
    rows: List[RowResult] = []
    for item in items:
        search_payload = await catalog_search(item.query_text, req.collection_name)
        points = search_payload.get("results", [])
        breakdown = (((search_payload.get("debug") or {}).get("score_trace") or {}).get("breakdown"))
        rows.append(map_row(item, points, req.probable_limit, breakdown=breakdown))

    should_build_spec = force_spec or len(items) > 2
    spec_file_id = write_spec_xlsx(rows) if should_build_spec else None

    reply_text = build_reply(rows, parser_meta, spec_file_id)
    if req.dialog_id:
        await append_short_memory(req.tenant_id, req.dialog_id, "user", req.message)
        await append_short_memory(req.tenant_id, req.dialog_id, "assistant", reply_text)

    return {
        "reply_text": reply_text,
        "items_count": len(items),
        "rows": [
            {
                "query_text": row.query_text,
                "replacement_article": row.replacement_article,
                "replacement_name": row.replacement_name,
                "score": row.score,
                "score_norm": row.score_norm,
                "price": row.price,
                "qty": row.qty,
                "total": row.total,
                "probable": row.probable,
                "dense_top": row.dense_top,
                "sparse_top": row.sparse_top,
                "hybrid_top": row.hybrid_top,
            }
            for row in rows
        ],
        "spec_file_id": spec_file_id,
        "spec_download_url": f"/agent/files/{spec_file_id}" if spec_file_id else None,
        "debug": {
            **parser_meta,
            "memory_enabled": bool(REDIS_URL),
            "memory_ttl_sec": MEMORY_TTL_SEC,
            "memory_dialog_id": req.dialog_id,
        },
    }


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/memory/reset")
async def memory_reset(req: MemoryResetRequest):
    deleted = await reset_short_memory(req.tenant_id, req.dialog_id)
    return {
        "status": "ok",
        "tenant_id": req.tenant_id,
        "dialog_id": req.dialog_id,
        "deleted": deleted,
    }


@app.post("/agent/chat")
async def agent_chat(req: AgentRequest):
    try:
        return await run_agent(req, force_spec=False)
    except HTTPException:
        raise
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail=f"Catalog API error: {exc}") from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/agent/spec")
async def agent_spec(req: AgentRequest):
    try:
        return await run_agent(req, force_spec=True)
    except HTTPException:
        raise
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail=f"Catalog API error: {exc}") from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/agent/files/{file_id}")
async def get_spec_file(file_id: str):
    safe_name = os.path.basename(file_id)
    file_path = SPEC_DIR / safe_name
    if not file_path.exists() or not file_path.is_file():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(
        path=file_path,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        filename=safe_name,
    )