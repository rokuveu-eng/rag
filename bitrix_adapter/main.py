#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
from typing import Any, Dict, Optional

import httpx
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field


ORCHESTRATOR_URL = os.getenv("ORCHESTRATOR_URL", "http://orchestrator:8430").rstrip("/")
DEFAULT_COLLECTION_NAME = os.getenv("DEFAULT_COLLECTION_NAME", "CHINT")
DEFAULT_TENANT_ID = os.getenv("DEFAULT_TENANT_ID", "default")

BITRIX_WEBHOOK_URL = (os.getenv("BITRIX_WEBHOOK_URL") or "").strip().rstrip("/")
BITRIX_BOT_ID = (os.getenv("BITRIX_BOT_ID") or "").strip()
BITRIX_CLIENT_ID = (os.getenv("BITRIX_CLIENT_ID") or "").strip()
BITRIX_WEBHOOK_VERIFY_TOKEN = (os.getenv("BITRIX_WEBHOOK_VERIFY_TOKEN") or "").strip()


app = FastAPI(title="Bitrix24 Adapter")


class TestSendRequest(BaseModel):
    dialog_id: str = Field(..., min_length=1)
    message: str = Field(..., min_length=1)


def _pick_path(data: Dict[str, Any], path: str) -> Optional[Any]:
    current: Any = data
    for part in path.split("."):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current.get(part)
    return current


def extract_dialog_id(payload: Dict[str, Any]) -> Optional[str]:
    for path in (
        "data.PARAMS.DIALOG_ID",
        "data.PARAMS.dialog_id",
        "data.DIALOG_ID",
        "DIALOG_ID",
        "dialog_id",
        "auth.user_id",
    ):
        value = _pick_path(payload, path)
        if value is not None and str(value).strip():
            return str(value).strip()
    return None


def extract_message_text(payload: Dict[str, Any]) -> str:
    for path in (
        "data.PARAMS.MESSAGE",
        "data.PARAMS.message",
        "data.MESSAGE",
        "MESSAGE",
        "message",
    ):
        value = _pick_path(payload, path)
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


def extract_tenant_id(payload: Dict[str, Any]) -> str:
    for path in ("auth.domain", "auth.member_id", "tenant_id"):
        value = _pick_path(payload, path)
        if value is not None and str(value).strip():
            return str(value).strip()
    return DEFAULT_TENANT_ID


def verify_token(request: Request, payload: Dict[str, Any]) -> bool:
    if not BITRIX_WEBHOOK_VERIFY_TOKEN:
        return True
    header_token = request.headers.get("x-webhook-token")
    query_token = request.query_params.get("token")
    body_token = str(payload.get("token") or "").strip() if isinstance(payload, dict) else ""
    return BITRIX_WEBHOOK_VERIFY_TOKEN in {header_token, query_token, body_token}


async def send_bitrix_message(dialog_id: str, message: str) -> Dict[str, Any]:
    if not BITRIX_WEBHOOK_URL:
        return {"status": "skipped", "reason": "BITRIX_WEBHOOK_URL is empty"}

    payload: Dict[str, Any] = {
        "DIALOG_ID": dialog_id,
        "MESSAGE": message,
    }
    if BITRIX_BOT_ID:
        payload["BOT_ID"] = int(BITRIX_BOT_ID)
    if BITRIX_CLIENT_ID:
        payload["CLIENT_ID"] = BITRIX_CLIENT_ID

    async with httpx.AsyncClient(timeout=20.0) as client:
        response = await client.post(f"{BITRIX_WEBHOOK_URL}/imbot.message.add", json=payload)
        response.raise_for_status()
        return response.json()


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/bitrix/send-test")
async def bitrix_send_test(req: TestSendRequest):
    try:
        result = await send_bitrix_message(req.dialog_id, req.message)
        return {"status": "ok", "result": result}
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Failed to send Bitrix message: {exc}") from exc


@app.post("/bitrix/webhook")
async def bitrix_webhook(request: Request):
    try:
        payload = await request.json()
    except Exception:
        payload = {}

    if not isinstance(payload, dict):
        payload = {"raw": payload}

    if not verify_token(request, payload):
        raise HTTPException(status_code=401, detail="Invalid webhook token")

    text = extract_message_text(payload)
    dialog_id = extract_dialog_id(payload)
    tenant_id = extract_tenant_id(payload)

    if not text:
        return {"status": "ignored", "reason": "empty_message"}

    if not dialog_id:
        return {"status": "ignored", "reason": "empty_dialog_id"}

    lower_text = text.strip().lower()
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            if lower_text == "/reset":
                reset_payload = {
                    "tenant_id": tenant_id,
                    "dialog_id": dialog_id,
                }
                reset_response = await client.post(
                    f"{ORCHESTRATOR_URL}/memory/reset",
                    json=reset_payload,
                )
                reset_response.raise_for_status()
                msg = "Память диалога очищена. Продолжаем с чистого контекста."
                sent = await send_bitrix_message(dialog_id, msg)
                return {
                    "status": "ok",
                    "action": "memory_reset",
                    "memory": reset_response.json(),
                    "sent": sent,
                }

            agent_payload = {
                "message": text,
                "collection_name": DEFAULT_COLLECTION_NAME,
                "tenant_id": tenant_id,
                "dialog_id": dialog_id,
            }
            agent_response = await client.post(f"{ORCHESTRATOR_URL}/agent/chat", json=agent_payload)
            agent_response.raise_for_status()
            agent_data = agent_response.json()

        reply_text = str(agent_data.get("reply_text") or "Не удалось сформировать ответ.")
        sent = await send_bitrix_message(dialog_id, reply_text)

        return {
            "status": "ok",
            "action": "chat",
            "dialog_id": dialog_id,
            "tenant_id": tenant_id,
            "items_count": agent_data.get("items_count"),
            "sent": sent,
        }
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail=f"Network error: {exc}") from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
