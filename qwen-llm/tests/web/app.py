from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import httpx
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse, Response, StreamingResponse
from pydantic import BaseModel, Field

from tests.web.reviews import WebReviewService


ROUTER_BASE_URL = "http://192.168.80.7:7100"


def _router_headers(request: Request) -> dict[str, str]:
    headers: dict[str, str] = {}
    for name, value in request.headers.items():
        if name.lower() in {
            "content-type",
            "x-gta-request-id",
            "x-gta-session-id",
            "x-gta-priority",
            "x-gta-locale",
            "x-gta-country",
        }:
            headers[name] = value
    return headers


class ChatDebugLog(BaseModel):
    event: Literal["用户发送", "模型返回"]
    content: str = Field(default="", max_length=100_000)
    reasoning: str = Field(default="", max_length=200_000)
    image_count: int = Field(default=0, ge=0, le=4)


def _index_html() -> str:
    return (Path(__file__).parent / "index.html").read_text(encoding="utf-8")


def create_web_router() -> APIRouter:
    router = APIRouter()
    review_service = WebReviewService()

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

    @router.post("/router/v1/chat/completions", include_in_schema=False)
    async def proxy_chat(request: Request) -> StreamingResponse:
        """让8080测试页通过同源地址流式访问本机7100 Router。"""
        client = httpx.AsyncClient(timeout=None)
        upstream_request = client.build_request(
            "POST",
            f"{ROUTER_BASE_URL}/v1/chat/completions",
            headers=_router_headers(request),
            content=await request.body(),
        )
        try:
            upstream = await client.send(upstream_request, stream=True)
        except Exception:
            await client.aclose()
            raise

        async def body_iterator():
            try:
                async for chunk in upstream.aiter_raw():
                    yield chunk
            finally:
                await upstream.aclose()
                await client.aclose()

        response_headers = {
            name: value
            for name, value in upstream.headers.items()
            if name.lower() in {"content-type", "cache-control", "x-gta-request-id"}
        }
        return StreamingResponse(
            body_iterator(),
            status_code=upstream.status_code,
            headers=response_headers,
        )

    @router.delete("/router/_gta/requests/{request_id}", include_in_schema=False)
    async def proxy_cancel(request_id: str, request: Request) -> Response:
        async with httpx.AsyncClient(timeout=10) as client:
            upstream = await client.delete(
                f"{ROUTER_BASE_URL}/_gta/requests/{request_id}",
                headers=_router_headers(request),
            )
        return Response(
            content=upstream.content,
            status_code=upstream.status_code,
            media_type=upstream.headers.get("content-type"),
        )

    @router.get("/api/reviews/summary", include_in_schema=False)
    async def review_summary(request: Request) -> dict[str, object]:
        locale = request.headers.get("x-gta-locale", "").strip() or "zh-CN"
        payload: dict[str, object] = {}
        try:
            google = await review_service.google(locale)
            payload["google"] = {key: value for key, value in google.items() if key != "reviews"}
        except (httpx.HTTPError, ValueError, KeyError):
            pass
        try:
            platform = await review_service.platform(page=1, page_size=1)
            payload["platform"] = {
                key: value for key, value in platform.items() if key != "reviews"
            }
        except (httpx.HTTPError, ValueError, KeyError):
            pass
        if not payload:
            raise HTTPException(status_code=503, detail="评价服务暂时不可用")
        return payload

    @router.get("/api/reviews", include_in_schema=False)
    async def reviews(
        request: Request,
        source: str,
        page: int = 1,
        page_size: int = 20,
    ) -> dict[str, object]:
        if page < 1 or page_size < 1 or page_size > 50:
            raise HTTPException(status_code=422, detail="page必须大于0, page_size范围为1至50")
        try:
            if source.strip().lower() == "google":
                locale = request.headers.get("x-gta-locale", "").strip() or "zh-CN"
                return await review_service.google(locale)
            if source.strip().lower() == "platform":
                return await review_service.platform(page=page, page_size=page_size)
        except (httpx.HTTPError, ValueError, KeyError) as exc:
            raise HTTPException(status_code=502, detail="评价上游服务调用失败") from exc
        raise HTTPException(status_code=422, detail="source只支持google或platform")

    return router
