from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from typing import Any

import pytest

from gta_ai.config import Settings
from gta_ai.media_audit import MediaAuditCoordinator, ProjectionNotReadyError
from gta_ai.schemas import (
    LocalGenerationResponse,
    MediaAuditRequest,
)


class FakeModel:
    async def generate(self, request: object) -> LocalGenerationResponse:
        return LocalGenerationResponse(
            model="audit-test-model",
            content=(
                '{"score":88,"hard_gate_passed":true,"confidence":0.96,'
                '"recommended_usage":"ARTICLE","review_opinion":"画面清晰且内容匹配",'
                '"strengths":["主体清楚"],"improvement_priorities":[],"issues":[]}'
            ),
        )


class FlakyModel(FakeModel):
    def __init__(self) -> None:
        self.calls = 0

    async def generate(self, request: object) -> LocalGenerationResponse:
        self.calls += 1
        if self.calls == 1:
            raise RuntimeError("temporary inference failure")
        return await super().generate(request)


class CapturingImageModel(FakeModel):
    def __init__(self) -> None:
        self.request: Any | None = None

    async def generate(self, request: object) -> LocalGenerationResponse:
        self.request = request
        return LocalGenerationResponse(
            model="audit-test-model",
            content=(
                '{"score":95,"hard_gate_passed":false,"confidence":0.95,'
                '"recommended_usage":"ARTICLE_INLINE","review_opinion":"古建筑画面清晰",'
                '"strengths":["主体清楚"],"improvement_priorities":[],'
                '"issues":[{"code":"SOURCE_MISSING","severity":"BLOCK",'
                '"scope":"IMAGE","start_ms":0,"end_ms":0,"evidence_ms":0,'
                '"confidence":0.9,"message":"无法验证来源","suggestion":"补充可信来源"}],'
                '"image_scores":{"clarity":1,"visible_rights_safety":100,'
                '"metadata_ai_usability":0,"duplicate_deduction":0},'
                '"recommended_title":"古宅门楼","image_suitability":'
                '{"approved_use_types":["ARTICLE_INLINE"]},'
                '"rights_assessment":{"risk_level":"LOW","reason":"无可见水印"}}'
            ),
        )


class FakeStore:
    def __init__(self, materials: dict[str, dict[str, Any]]) -> None:
        self.materials = materials
        self.results: dict[str, dict[str, Any]] = {}

    async def get_material(self, asset_id: str) -> dict[str, Any]:
        return self.materials[asset_id]

    async def get_result(self, asset_id: str) -> dict[str, Any] | None:
        return self.results.get(asset_id)

    async def put_result(self, asset_id: str, document: dict[str, Any]) -> None:
        self.results[asset_id] = document


class FakeCallback:
    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []

    async def send(self, event: dict[str, Any]) -> None:
        self.events.append(event)


class DelayedStore(FakeStore):
    def __init__(self, material: dict[str, Any]) -> None:
        super().__init__({"3": material})
        self.reads = 0

    async def get_material(self, asset_id: str) -> dict[str, Any]:
        self.reads += 1
        if self.reads == 1:
            raise ProjectionNotReadyError("projection pending")
        return await super().get_material(asset_id)


def request(asset_id: str, request_id: str, media_type: str) -> MediaAuditRequest:
    return MediaAuditRequest(
        request_id=request_id,
        asset_id=asset_id,
        asset_generation=f"generation-{asset_id}",
        media_type=media_type,  # type: ignore[arg-type]
        source_key=f"{media_type}/{asset_id}.bin",
        source_md5="0123456789abcdef0123456789abcdef",
        audit_input_hash=(asset_id * 64)[:64],
        policy_version="media-audit-v5",
        requested_at=datetime.now(UTC),
    )


@pytest.mark.asyncio
async def test_one_image_and_one_video_complete_without_database_access() -> None:
    requests = [
        request("1", "request-image-1234", "image"),
        request("2", "request-video-1234", "video"),
    ]
    materials = {
        current.asset_id: {
            "asset_id": current.asset_id,
            "asset_generation": current.asset_generation,
            "current_audit_request_id": current.request_id,
            "deletion_state": "ACTIVE",
            "analysis": {"status": "READY", "analysis_version": "analysis-v1"},
            "image_analysis": {"analysis_status": "READY"},
            "media_type": current.media_type,
            "title": f"sample-{current.media_type}",
        }
        for current in requests
    }
    store = FakeStore(materials)
    callback = FakeCallback()
    settings = Settings.for_tests(
        media_audit_enabled=True,
        media_audit_api_token="token",
        media_audit_java_callback_url="http://java/callback",
        media_audit_java_callback_secret="secret",
    )
    coordinator = MediaAuditCoordinator(
        settings,
        FakeModel(),  # type: ignore[arg-type]
        store=store,  # type: ignore[arg-type]
        callback=callback,  # type: ignore[arg-type]
    )
    await coordinator.start()
    try:
        for current in requests:
            await coordinator.submit(current)
        for _ in range(100):
            states = [await coordinator.status(current.request_id) for current in requests]
            if all(state.state == "completed" for state in states):
                break
            await asyncio.sleep(0.01)
        final_states = [await coordinator.status(current.request_id) for current in requests]
        assert all(state.state == "completed" for state in final_states)
    finally:
        await coordinator.close()

    assert set(store.results) == {"1", "2"}
    assert {event["media_type"] for event in callback.events} == {"image", "video"}
    assert all(event["decision"] == "PASS" for event in callback.events)
    image_event = next(event for event in callback.events if event["media_type"] == "image")
    assert image_event["image_scores"]["clarity_contribution"] == 48.0
    assert image_event["image_scores"]["visible_rights_contribution"] == 40.0
    assert image_event["image_scores"]["metadata_ai_usability"] == 100.0
    assert image_event["score"] == 88.0


@pytest.mark.asyncio
async def test_audit_waits_for_projection_without_marking_request_failed() -> None:
    current = request("3", "request-wait-12345", "image")
    material = {
        "asset_id": current.asset_id,
        "asset_generation": current.asset_generation,
        "current_audit_request_id": current.request_id,
        "deletion_state": "ACTIVE",
        "analysis": {"status": "READY", "analysis_version": "analysis-v1"},
        "image_analysis": {"analysis_status": "READY"},
        "media_type": "image",
    }
    store = DelayedStore(material)
    callback = FakeCallback()
    runtime = Settings.for_tests(
        media_audit_enabled=True,
        media_audit_api_token="token",
        media_audit_java_callback_url="http://java/callback",
        media_audit_java_callback_secret="secret",
        media_audit_projection_retry_seconds=0.01,
    )
    coordinator = MediaAuditCoordinator(
        runtime,
        FakeModel(),  # type: ignore[arg-type]
        store=store,  # type: ignore[arg-type]
        callback=callback,  # type: ignore[arg-type]
    )
    await coordinator.start()
    try:
        await coordinator.submit(current)
        for _ in range(100):
            state = await coordinator.status(current.request_id)
            if state.state == "completed":
                break
            assert state.state != "failed"
            await asyncio.sleep(0.01)
    finally:
        await coordinator.close()

    assert store.reads >= 2
    assert len(callback.events) == 1


@pytest.mark.asyncio
async def test_transient_model_failure_is_retried_without_java_resubmit() -> None:
    current = request("4", "request-retry-1234", "image")
    material = {
        "asset_id": current.asset_id,
        "asset_generation": current.asset_generation,
        "current_audit_request_id": current.request_id,
        "deletion_state": "ACTIVE",
        "analysis": {"status": "READY", "analysis_version": "analysis-v1"},
        "image_analysis": {"analysis_status": "READY"},
        "media_type": "image",
    }
    store = FakeStore({current.asset_id: material})
    callback = FakeCallback()
    model = FlakyModel()
    runtime = Settings.for_tests(
        media_audit_enabled=True,
        media_audit_api_token="token",
        media_audit_java_callback_url="http://java/callback",
        media_audit_java_callback_secret="secret",
        media_audit_transient_retry_max_attempts=1,
        media_audit_transient_retry_seconds=0.01,
    )
    coordinator = MediaAuditCoordinator(
        runtime,
        model,  # type: ignore[arg-type]
        store=store,  # type: ignore[arg-type]
        callback=callback,  # type: ignore[arg-type]
    )
    await coordinator.start()
    try:
        await coordinator.submit(current)
        for _ in range(100):
            state = await coordinator.status(current.request_id)
            if state.state == "completed":
                break
            assert state.state != "failed"
            await asyncio.sleep(0.01)
        assert (await coordinator.status(current.request_id)).state == "completed"
    finally:
        await coordinator.close()

    assert model.calls == 2
    assert len(callback.events) == 1


@pytest.mark.asyncio
async def test_failed_p1_can_be_resubmitted_as_deferred_p9() -> None:
    current = request("6", "request-deferred-retry", "image")
    material = {
        "asset_id": current.asset_id,
        "asset_generation": current.asset_generation,
        "current_audit_request_id": current.request_id,
        "deletion_state": "ACTIVE",
        "analysis": {"status": "READY", "analysis_version": "analysis-v1"},
        "image_analysis": {"analysis_status": "READY"},
        "media_type": "image",
    }
    model = FlakyModel()
    callback = FakeCallback()
    coordinator = MediaAuditCoordinator(
        Settings.for_tests(
            media_audit_enabled=True,
            media_audit_api_token="token",
            media_audit_java_callback_url="http://java/callback",
            media_audit_java_callback_secret="secret",
            media_audit_transient_retry_max_attempts=0,
        ),
        model,  # type: ignore[arg-type]
        store=FakeStore({current.asset_id: material}),  # type: ignore[arg-type]
        callback=callback,  # type: ignore[arg-type]
    )
    await coordinator.start()
    try:
        await coordinator.submit(current)
        for _ in range(100):
            if (await coordinator.status(current.request_id)).state == "failed":
                break
            await asyncio.sleep(0.01)
        assert (await coordinator.status(current.request_id)).state == "failed"

        await coordinator.submit(current.model_copy(update={"priority": "P9"}))
        for _ in range(100):
            if (await coordinator.status(current.request_id)).state == "completed":
                break
            await asyncio.sleep(0.01)
        assert (await coordinator.status(current.request_id)).state == "completed"
    finally:
        await coordinator.close()

    assert model.calls == 2
    assert [event["event_type"] for event in callback.events] == [
        "material.ai.audit.failed",
        "material.ai.audit.completed",
    ]


@pytest.mark.asyncio
async def test_image_uses_compact_ready_evidence_and_objective_quality() -> None:
    current = request("5", "request-image-fast", "image")
    material = {
        "asset_id": current.asset_id,
        "asset_generation": current.asset_generation,
        "current_audit_request_id": current.request_id,
        "deletion_state": "ACTIVE",
        "analysis": {"status": "READY", "analysis_version": "analysis-v1"},
        "image_analysis": {
            "analysis_status": "READY",
            "analysis_version": "image-analysis-v1",
            "quality": {"score": 92, "resolution_passed": True},
            "semantic": {"description": "古镇木结构门楼"},
            "vision_embedding": {"vector": [0.1] * 768},
        },
        "media_type": "image",
        "title": "古宅门楼",
    }
    store = FakeStore({current.asset_id: material})
    callback = FakeCallback()
    model = CapturingImageModel()
    runtime = Settings.for_tests(
        media_audit_enabled=True,
        media_audit_api_token="token",
        media_audit_java_callback_url="http://java/callback",
        media_audit_java_callback_secret="secret",
        media_audit_image_max_output_tokens=700,
    )
    coordinator = MediaAuditCoordinator(
        runtime,
        model,  # type: ignore[arg-type]
        store=store,  # type: ignore[arg-type]
        callback=callback,  # type: ignore[arg-type]
    )
    await coordinator.start()
    try:
        await coordinator.submit(current)
        for _ in range(100):
            if (await coordinator.status(current.request_id)).state == "completed":
                break
            await asyncio.sleep(0.01)
    finally:
        await coordinator.close()

    assert model.request is not None
    assert model.request.max_tokens == 700
    assert model.request.enable_thinking is False
    assert model.request.response_format["type"] == "json_schema"
    assert "vision_embedding" not in model.request.messages[-1].content
    result = store.results[current.asset_id]
    assert result["image_scores"]["clarity"] == 92
    assert result["image_scores"]["metadata_ai_usability"] == 100
    assert result["issues"] == []
    assert result["decision"] == "PASS"
