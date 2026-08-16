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
async def test_internal_inference_requires_credential() -> None:
    app = create_app(
        settings=Settings.for_tests(internal_inference_api_token="worker-token"),
        local_model_client=FakeLocalModelClient(),  # type: ignore[arg-type]
    )
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post("/v1/chat/completions", json={})
    assert response.status_code == 401


@pytest.mark.asyncio
async def test_internal_inference_forwards_openai_payload_and_priority() -> None:
    async def upstream(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/chat/completions"
        assert request.headers["x-gta-priority"] == "P9"
        assert json.loads(request.content)["model"] == "test-model"
        return httpx.Response(
            200,
            json={
                "model": "test-model",
                "choices": [{"message": {"content": "{}"}}],
            },
        )

    app = create_app(
        settings=Settings.for_tests(
            internal_inference_api_token="worker-token",
            local_llm_model="test-model",
        ),
        local_model_client=FakeLocalModelClient(),  # type: ignore[arg-type]
        inference_transport=httpx.MockTransport(upstream),
    )
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post(
            "/v1/chat/completions",
            headers={
                "Authorization": "Bearer worker-token",
                "X-GTA-Priority": "P9",
            },
            json={"model": "test-model", "messages": []},
        )
    assert response.status_code == 200
    assert response.json()["model"] == "test-model"


@pytest.mark.asyncio
async def test_internal_inference_normalizes_legacy_priority_header() -> None:
    async def upstream(request: httpx.Request) -> httpx.Response:
        assert request.headers["x-gta-priority"] == "P1"
        assert "x-gta-workload-priority" not in request.headers
        return httpx.Response(
            200, json={"model": "test-model", "choices": [{"message": {"content": "{}"}}]}
        )

    app = create_app(
        settings=Settings.for_tests(
            internal_inference_api_token="worker-token",
            local_llm_model="test-model",
        ),
        local_model_client=FakeLocalModelClient(),  # type: ignore[arg-type]
        inference_transport=httpx.MockTransport(upstream),
    )
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post(
            "/v1/chat/completions",
            headers={
                "Authorization": "Bearer worker-token",
                "X-GTA-Workload-Priority": "P1",
            },
            json={"model": "test-model", "messages": []},
        )
    assert response.status_code == 200
