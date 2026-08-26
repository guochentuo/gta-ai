from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from fastapi import APIRouter
from fastapi.responses import HTMLResponse, Response
from pydantic import BaseModel, Field


class ChatDebugLog(BaseModel):
    event: Literal["用户发送", "模型返回"]
    content: str = Field(default="", max_length=100_000)
    reasoning: str = Field(default="", max_length=200_000)
    image_count: int = Field(default=0, ge=0, le=4)


def _index_html() -> str:
    return (Path(__file__).parent / "index.html").read_text(encoding="utf-8")


def create_web_router() -> APIRouter:
    router = APIRouter()

    @router.get("/", response_class=HTMLResponse, include_in_schema=False)
    async def index() -> HTMLResponse:
        return HTMLResponse(_index_html(), headers={"Cache-Control": "no-store"})

    @router.get("/favicon.ico", include_in_schema=False)
    async def favicon() -> Response:
        return Response(status_code=204)

    @router.post("/debug/chat-log", include_in_schema=False)
    async def chat_log(entry: ChatDebugLog) -> Response:
        payload = entry.model_dump()
        print(
            f"[GTA AI Web] {entry.event} "
            f"{json.dumps(payload, ensure_ascii=False, separators=(',', ':'))}",
            flush=True,
        )
        return Response(status_code=204)

    return router
