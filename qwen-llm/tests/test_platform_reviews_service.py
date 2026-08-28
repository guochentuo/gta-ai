from __future__ import annotations

import httpx
import pytest

from tests.web.reviews import WebReviewService


@pytest.mark.asyncio
async def test_only_approved_reviews_are_exposed() -> None:
    requested_status: str | None = None

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal requested_status
        if request.url.path.endswith("/_search"):
            return httpx.Response(
                200,
                json={
                    "hits": {
                        "hits": [
                            {
                                "_source": {
                                    "engagement": {
                                        "rating_score": 5,
                                        "rating_count": 1,
                                    }
                                }
                            }
                        ]
                    }
                },
            )
        if request.url.path == "/review/list":
            requested_status = request.url.params.get("status")
            return httpx.Response(
                200,
                json={
                    "rows": [
                        {"bizId": 1, "status": 1, "star": 5, "description": "已审核"},
                        {"bizId": 2, "status": 0, "star": 5, "description": "待审核"},
                        {"bizId": 3, "status": 2, "star": 1, "description": "已拒绝"},
                    ]
                },
            )
        if request.url.path.endswith("/_mget"):
            return httpx.Response(
                200,
                json={
                    "docs": [
                        {
                            "_id": "trip:1",
                            "found": True,
                            "_source": {"title": "审核通过的行程", "display_url": "trip/1"},
                        }
                    ]
                },
            )
        raise AssertionError(f"unexpected request: {request.method} {request.url}")

    service = WebReviewService(transport=httpx.MockTransport(handler))

    result = await service.platform()

    assert requested_status == "1"
    assert [review["text"] for review in result["reviews"]] == ["已审核"]
