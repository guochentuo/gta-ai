from __future__ import annotations

import json
from collections.abc import AsyncIterator
from pathlib import Path

import httpx
from fastapi import APIRouter
from fastapi.responses import HTMLResponse, Response, StreamingResponse

from gta_ai.config import Settings
from gta_ai.schemas.chat import BrowserChatRequest


def _index_html() -> str:
    return (Path(__file__).parent / "web" / "index.html").read_text(encoding="utf-8")


def create_web_router(
    settings: Settings,
    *,
    transport: httpx.AsyncBaseTransport | None = None,
) -> APIRouter:
    router = APIRouter()

    @router.get("/", response_class=HTMLResponse, include_in_schema=False)
    async def index() -> HTMLResponse:
        return HTMLResponse(_index_html(), headers={"Cache-Control": "no-store"})

    @router.get("/favicon.ico", include_in_schema=False)
    async def favicon() -> Response:
        return Response(status_code=204)

    @router.post("/api/chat", include_in_schema=False)
    async def chat(request: BrowserChatRequest) -> StreamingResponse:
        payload = {
            "model": settings.local_llm_model,
            "messages": [message.model_dump(mode="json") for message in request.messages],
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
            "stream": True,
            "chat_template_kwargs": {"enable_thinking": request.enable_thinking},
        }

        async def proxy_stream() -> AsyncIterator[str]:
            timeout = httpx.Timeout(
                connect=10,
                read=settings.local_llm_timeout_seconds,
                write=30,
                pool=10,
            )
            headers = {"Accept": "text/event-stream"}
            if settings.local_llm_api_key is not None:
                token = settings.local_llm_api_key.get_secret_value()
                headers["Authorization"] = f"Bearer {token}"
            try:
                async with httpx.AsyncClient(
                    timeout=timeout,
                    transport=transport,
                    headers=headers,
                ) as client, client.stream(
                    "POST",
                    settings.local_llm_url("chat/completions"),
                    json=payload,
                ) as response:
                    if response.status_code >= 400:
                        detail = (await response.aread()).decode(errors="replace")[:1000]
                        message = (
                            f"Local model returned HTTP {response.status_code}: {detail}"
                        )
                        error = json.dumps({"error": message})
                        yield f"data: {error}\n\n"
                        yield "data: [DONE]\n\n"
                        return

                    async for line in response.aiter_lines():
                        if line.startswith("data:"):
                            yield f"{line}\n\n"
            except httpx.HTTPError as exc:
                error = json.dumps(
                    {"error": f"Local model connection failed: {type(exc).__name__}"}
                )
                yield f"data: {error}\n\n"
                yield "data: [DONE]\n\n"

        return StreamingResponse(
            proxy_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache, no-transform",
                "X-Accel-Buffering": "no",
            },
        )

    return router
