from __future__ import annotations

import httpx
import pytest

from gta_ai.api import create_app
from gta_ai.config import Settings
from gta_ai.schemas import HealthState, ProviderHealth


class FakeLocalModelClient:
    async def probe(self) -> ProviderHealth:
        return ProviderHealth(provider="local_llm", state=HealthState.OK, model="test")


class FakeAuditWorker:
    async def start(self) -> None:
        return None

    async def close(self) -> None:
        return None


@pytest.mark.asyncio
async def test_http_audit_submission_contract_was_removed() -> None:
    app = create_app(
        settings=Settings.for_tests(),
        local_model_client=FakeLocalModelClient(),  # type: ignore[arg-type]
        media_audit_worker=FakeAuditWorker(),
    )
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post("/v1/media-audits", json={})

    assert response.status_code == 404
