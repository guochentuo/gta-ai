from __future__ import annotations

from time import perf_counter

import httpx

from gta_ai.config import Settings
from gta_ai.schemas import (
    HealthState,
    LocalGenerationRequest,
    LocalGenerationResponse,
    ProviderHealth,
)


class LocalModelError(RuntimeError):
    pass


class LocalModelClient:
    """Small client for the OpenAI-compatible API exposed by vLLM."""

    def __init__(
        self,
        settings: Settings,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._settings = settings
        self._transport = transport

    def _headers(self, priority: str | None = None) -> dict[str, str]:
        headers: dict[str, str] = {}
        if self._settings.local_llm_api_key is not None:
            headers["Authorization"] = (
                f"Bearer {self._settings.local_llm_api_key.get_secret_value()}"
            )
        if priority:
            headers["X-GTA-Priority"] = priority
        return headers

    async def probe(self) -> ProviderHealth:
        started = perf_counter()
        try:
            timeout = httpx.Timeout(self._settings.local_llm_health_timeout_seconds)
            async with httpx.AsyncClient(
                timeout=timeout,
                transport=self._transport,
                headers=self._headers(),
            ) as client:
                response = await client.get(self._settings.local_llm_url("models"))
                response.raise_for_status()
                payload = response.json()

            available_models = {
                str(item.get("id"))
                for item in payload.get("data", [])
                if isinstance(item, dict) and item.get("id")
            }
            target_available = self._settings.local_llm_model in available_models
            return ProviderHealth(
                provider="local_llm",
                state=HealthState.OK if target_available else HealthState.DEGRADED,
                model=self._settings.local_llm_model,
                latency_ms=(perf_counter() - started) * 1000,
                detail=None if target_available else "configured model is not advertised",
            )
        except (httpx.HTTPError, ValueError, TypeError) as exc:
            return ProviderHealth(
                provider="local_llm",
                state=HealthState.UNAVAILABLE,
                model=self._settings.local_llm_model,
                latency_ms=(perf_counter() - started) * 1000,
                detail=type(exc).__name__,
            )

    async def generate(self, request: LocalGenerationRequest) -> LocalGenerationResponse:
        payload = {
            "model": self._settings.local_llm_model,
            "messages": [message.model_dump(mode="json") for message in request.messages],
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
            "chat_template_kwargs": {"enable_thinking": request.enable_thinking},
        }
        if request.response_format is not None:
            payload["response_format"] = request.response_format
        try:
            timeout = httpx.Timeout(self._settings.local_llm_timeout_seconds)
            async with httpx.AsyncClient(
                timeout=timeout,
                transport=self._transport,
                headers=self._headers(request.priority),
            ) as client:
                response = await client.post(
                    self._settings.local_llm_url("chat/completions"), json=payload
                )
                response.raise_for_status()
                body = response.json()

            choice = body["choices"][0]
            content = choice["message"]["content"]
            if not isinstance(content, str):
                raise TypeError("local model returned non-text content")
            usage = body.get("usage") if isinstance(body.get("usage"), dict) else {}
            return LocalGenerationResponse(
                model=str(body.get("model") or self._settings.local_llm_model),
                content=content,
                finish_reason=(
                    str(choice["finish_reason"])
                    if choice.get("finish_reason") is not None
                    else None
                ),
                prompt_tokens=usage.get("prompt_tokens"),
                completion_tokens=usage.get("completion_tokens"),
                total_tokens=usage.get("total_tokens"),
            )
        except (httpx.HTTPError, KeyError, IndexError, TypeError, ValueError) as exc:
            raise LocalModelError(f"local model request failed: {type(exc).__name__}") from exc
