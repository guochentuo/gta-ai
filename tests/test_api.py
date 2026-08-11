import httpx
import pytest

from gta_ai.api import create_app
from gta_ai.config import Settings
from gta_ai.schemas import HealthState, ProviderHealth


class FakeLocalModelClient:
    def __init__(self, state: HealthState) -> None:
        self.state = state

    async def probe(self) -> ProviderHealth:
        return ProviderHealth(
            provider="local_llm",
            state=self.state,
            model="Qwen/Qwen3.6-27B-FP8",
        )


@pytest.mark.asyncio
async def test_liveness_never_requires_a_model() -> None:
    app = create_app(
        settings=Settings.for_tests(),
        local_model_client=FakeLocalModelClient(HealthState.UNAVAILABLE),  # type: ignore[arg-type]
    )

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/health/live")

    assert response.status_code == 200
    assert response.json()["state"] == "ok"


@pytest.mark.asyncio
async def test_readiness_is_503_until_local_model_is_available() -> None:
    app = create_app(
        settings=Settings.for_tests(),
        local_model_client=FakeLocalModelClient(HealthState.UNAVAILABLE),  # type: ignore[arg-type]
    )

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/health/ready")

    assert response.status_code == 503
    assert response.json()["state"] == "degraded"


@pytest.mark.asyncio
async def test_readiness_is_ok_with_local_model() -> None:
    app = create_app(
        settings=Settings.for_tests(openai_api_key="configured"),
        local_model_client=FakeLocalModelClient(HealthState.OK),  # type: ignore[arg-type]
    )

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/health/ready")

    assert response.status_code == 200
    assert response.json()["providers"][1]["state"] == "configured"
