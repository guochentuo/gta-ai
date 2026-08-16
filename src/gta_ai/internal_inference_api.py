from __future__ import annotations

import hmac

import httpx
from fastapi import APIRouter, Header, HTTPException, Request, Response

from gta_ai.config import Settings


def create_internal_inference_router(
    settings: Settings,
    *,
    transport: httpx.AsyncBaseTransport | None = None,
) -> APIRouter:
    """Authenticated OpenAI-compatible bridge for internal non-Java workers."""

    router = APIRouter(prefix="/v1", tags=["internal-inference"])

    def authorize(authorization: str | None) -> None:
        # Test and small deployments may intentionally share the existing internal
        # media token. A dedicated token takes precedence when one is configured.
        secret = (
            settings.internal_inference_api_token or settings.media_audit_api_token
        )
        if secret is None:
            raise HTTPException(
                status_code=503, detail="internal inference token is not configured"
            )
        expected = f"Bearer {secret.get_secret_value()}"
        if authorization is None or not hmac.compare_digest(authorization, expected):
            raise HTTPException(status_code=401, detail="invalid inference credential")

    @router.post("/chat/completions")
    async def chat_completions(
        request: Request,
        authorization: str | None = Header(default=None),
        priority: str | None = Header(default=None, alias="X-GTA-Priority"),
        legacy_workload_priority: str | None = Header(
            default=None, alias="X-GTA-Workload-Priority"
        ),
    ) -> Response:
        authorize(authorization)
        headers = {"Content-Type": "application/json"}
        # worker、OCR、ASR 和推理路由统一使用 X-GTA-Priority。旧名称只
        # 作为兼容输入, 转发时必须规范化, 否则 P9 会被路由器误当成 P1.
        normalized_priority = priority or legacy_workload_priority
        if normalized_priority:
            headers["X-GTA-Priority"] = normalized_priority
        if settings.local_llm_api_key is not None:
            headers["Authorization"] = (
                f"Bearer {settings.local_llm_api_key.get_secret_value()}"
            )
        try:
            timeout = httpx.Timeout(
                connect=10,
                read=settings.local_llm_timeout_seconds,
                write=30,
                pool=10,
            )
            async with httpx.AsyncClient(
                timeout=timeout,
                transport=transport,
                headers=headers,
            ) as client:
                upstream = await client.post(
                    settings.local_llm_url("chat/completions"),
                    content=await request.body(),
                )
        except httpx.HTTPError as exc:
            raise HTTPException(
                status_code=503,
                detail=f"local inference unavailable: {type(exc).__name__}",
            ) from exc

        response_headers: dict[str, str] = {}
        content_type = upstream.headers.get("content-type")
        if content_type:
            response_headers["Content-Type"] = content_type
        return Response(
            content=upstream.content,
            status_code=upstream.status_code,
            headers=response_headers,
        )

    return router
