from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from gta_ai.api import create_app
from gta_ai.config import Settings
from gta_ai.material_search import ElasticsearchMaterialSearch
from gta_ai.schemas import (
    HealthState,
    MaterialSearchRequest,
    MaterialSearchResult,
    ProviderHealth,
)


class FakeLocalModelClient:
    async def probe(self) -> ProviderHealth:
        return ProviderHealth(provider="local_llm", state=HealthState.OK, model="test")


class FakeAuditCoordinator:
    async def start(self) -> None:
        return None

    async def close(self) -> None:
        return None


class FakeSearch:
    def __init__(self) -> None:
        self.requests: list[MaterialSearchRequest] = []

    async def search(self, request: MaterialSearchRequest) -> MaterialSearchResult:
        self.requests.append(request)
        return MaterialSearchResult(total=1, list=[{"id": "42", "originName": "乌镇夜景"}])


def settings() -> Settings:
    return Settings.for_tests(
        media_audit_api_token="shared-token",
        material_search_enabled=True,
        material_search_embedding_enabled=False,
    )


@pytest.mark.asyncio
async def test_material_search_api_requires_bearer_and_forwards_filters() -> None:
    search = FakeSearch()
    app = create_app(
        settings=settings(),
        local_model_client=FakeLocalModelClient(),  # type: ignore[arg-type]
        media_audit_worker=FakeAuditCoordinator(),  # type: ignore[arg-type]
        material_search_service=search,
    )
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        denied = await client.post("/v1/material-search", json={"keyword": "乌镇"})
        accepted = await client.post(
            "/v1/material-search",
            json={"keyword": "乌镇", "type": "02", "page_num": 2, "page_size": 12},
            headers={"Authorization": "Bearer shared-token"},
        )

    assert denied.status_code == 401
    assert accepted.status_code == 200
    assert accepted.json()["list"][0]["originName"] == "乌镇夜景"
    assert search.requests[0].type == "02"
    assert search.requests[0].page_num == 2


@pytest.mark.asyncio
async def test_es_search_keeps_vector_candidates_inside_lexical_filter() -> None:
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured.update(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "hits": {
                    "total": {"value": 1},
                    "hits": [
                        {
                            "_source": {
                                "asset_id": "9",
                                "media_type": "image",
                                "origin_name": "乌镇河道",
                                "deletion_state": "ACTIVE",
                                "created_by": "7",
                                "created_at": "2026-08-15T10:20:30Z",
                                "updated_at": "2026-08-15T10:21:30Z",
                                "size_bytes": 2048,
                                "metrics": {"quote_count": 0},
                            }
                        }
                    ],
                }
            },
        )

    service = ElasticsearchMaterialSearch(
        settings(), transport=httpx.MockTransport(handler)
    )
    result = await service.search(MaterialSearchRequest(keyword="乌镇"))

    assert captured["query"]["bool"]["must"][0]["bool"]["minimum_should_match"] == 1
    assert "knn" not in captured
    assert result.total == 1
    assert result.list[0]["sizeTxt"] == "2.00 KB"
    assert result.list[0]["createdAt"] == "2026-08-15 10:20:30"
