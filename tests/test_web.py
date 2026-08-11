from __future__ import annotations

import json

import httpx
import pytest

from gta_ai.api import create_app
from gta_ai.config import Settings
from gta_ai.schemas import HealthState, ProviderHealth


class FakeLocalModelClient:
    async def probe(self) -> ProviderHealth:
        return ProviderHealth(
            provider="local_llm",
            state=HealthState.OK,
            model="Qwen/Qwen3.6-27B-FP8",
        )


@pytest.mark.asyncio
async def test_index_serves_chat_interface() -> None:
    app = create_app(
        settings=Settings.for_tests(),
        local_model_client=FakeLocalModelClient(),  # type: ignore[arg-type]
    )

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.get("/")

    assert response.status_code == 200
    assert response.headers["cache-control"] == "no-store"
    assert "GTA AI" in response.text
    assert 'fetch("/api/chat"' in response.text
    assert '<details class="reasoning" open>' in response.text
    assert "scrollbar-gutter: stable" in response.text
    assert "grid-template-rows: 56px minmax(0, 1fr) auto" in response.text


@pytest.mark.asyncio
async def test_chat_proxy_forwards_request_and_streams_sse() -> None:
    async def upstream(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        assert request.url.path == "/v1/chat/completions"
        assert request.headers["authorization"] == "Bearer test-token"
        assert payload["model"] == "Qwen/Qwen3.6-27B-FP8"
        assert payload["stream"] is True
        assert payload["chat_template_kwargs"] == {"enable_thinking": False}
        body = (
            'data: {"choices":[{"delta":{"content":"你好"}}]}\n\n'
            "data: [DONE]\n\n"
        )
        return httpx.Response(200, text=body, headers={"content-type": "text/event-stream"})

    app = create_app(
        settings=Settings.for_tests(local_llm_api_key="test-token"),
        local_model_client=FakeLocalModelClient(),  # type: ignore[arg-type]
        chat_transport=httpx.MockTransport(upstream),
    )

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post(
            "/api/chat",
            json={
                "messages": [{"role": "user", "content": "你好"}],
                "enable_thinking": False,
                "max_tokens": 256,
                "temperature": 0.2,
            },
        )

    assert response.status_code == 200
    assert 'data: {"choices"' in response.text
    assert "data: [DONE]" in response.text


@pytest.mark.asyncio
async def test_chat_proxy_rejects_unknown_request_fields() -> None:
    app = create_app(
        settings=Settings.for_tests(),
        local_model_client=FakeLocalModelClient(),  # type: ignore[arg-type]
    )

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post(
            "/api/chat",
            json={
                "messages": [{"role": "user", "content": "你好"}],
                "unexpected": True,
            },
        )

    assert response.status_code == 422
