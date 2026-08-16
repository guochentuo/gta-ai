# ruff: noqa: RUF001, RUF002, RUF003 -- 审核提示词和说明使用中文全角标点。

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import logging
import re
import time
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Protocol

import httpx

from gta_ai.clients.local_model import LocalModelClient
from gta_ai.config import Settings
from gta_ai.schemas import (
    LocalGenerationRequest,
    MediaAuditAccepted,
    MediaAuditRequest,
    MediaAuditState,
    ModelMessage,
)

_AUDIT_LOGGER = logging.getLogger("gta_ai.media_audit")


def _configure_audit_log(path: str) -> None:
    if not path.strip():
        return
    resolved = Path(path).expanduser().resolve()
    if any(
        getattr(handler, "gta_ai_audit_path", None) == resolved
        for handler in _AUDIT_LOGGER.handlers
    ):
        return
    resolved.parent.mkdir(parents=True, exist_ok=True)
    handler = RotatingFileHandler(
        resolved,
        maxBytes=20 * 1024 * 1024,
        backupCount=10,
        encoding="utf-8",
    )
    handler.setFormatter(logging.Formatter("%(message)s"))
    handler.gta_ai_audit_path = resolved  # type: ignore[attr-defined]
    _AUDIT_LOGGER.addHandler(handler)
    _AUDIT_LOGGER.setLevel(logging.INFO)
    _AUDIT_LOGGER.propagate = False


def _audit_log(event: str, *, level: int = logging.INFO, **fields: object) -> None:
    """输出可检索的一行 JSON，不记录提示词、密钥或原始素材内容。"""
    _AUDIT_LOGGER.log(
        level,
        json.dumps(
            {
                "timestamp": datetime.now(UTC)
                .isoformat(timespec="milliseconds")
                .replace("+00:00", "Z"),
                "level": logging.getLevelName(level),
                "service": "gta-ai",
                "module": "MediaAudit",
                "event": event,
                **fields,
            },
            ensure_ascii=False,
            separators=(",", ":"),
        ),
    )


class MediaAuditError(RuntimeError):
    pass


class ProjectionNotReadyError(MediaAuditError):
    """ES 投影仍在最终一致性窗口内；请求必须等待，不应被判为审核失败。"""


class MediaAuditCoordinatorPort(Protocol):
    async def start(self) -> None: ...

    async def close(self) -> None: ...

    async def submit(self, request: MediaAuditRequest) -> MediaAuditAccepted: ...

    async def cancel(self, request_id: str) -> MediaAuditState: ...

    async def status(self, request_id: str) -> MediaAuditState: ...


def _utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _audit_id(asset_id: str) -> str:
    # 与既有 V1 稳定文档 ID 保持一致；重新审核覆盖，不创建 V2/V3 索引版本。
    value = f"media-content-audit-v1:{asset_id}".encode()
    return hashlib.sha256(value).hexdigest()


def _event_id(request_id: str, completed_at: str) -> str:
    value = f"{request_id}:{completed_at}:{time.time_ns()}".encode()
    return hashlib.sha256(value).hexdigest()


def _bounded_text(value: object, maximum: int) -> str:
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
    return text if len(text) <= maximum else text[:maximum] + "…"


def _extract_json_object(text: str) -> dict[str, Any]:
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.IGNORECASE)
    try:
        value = json.loads(cleaned)
    except json.JSONDecodeError:
        begin = cleaned.find("{")
        end = cleaned.rfind("}")
        if begin < 0 or end <= begin:
            raise MediaAuditError("model did not return a JSON object") from None
        try:
            value = json.loads(cleaned[begin : end + 1])
        except json.JSONDecodeError as exc:
            raise MediaAuditError("model returned invalid JSON") from exc
    if not isinstance(value, dict):
        raise MediaAuditError("model result must be an object")
    return value


_ISSUE_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "code": {"type": "string"},
        "severity": {"type": "string", "enum": ["BLOCK", "ERROR", "WARN"]},
        "scope": {
            "type": "string",
            "enum": ["GLOBAL", "TIMELINE", "IMAGE", "DELIVERY_CONFIG"],
        },
        "start_ms": {"type": "integer", "minimum": 0},
        "end_ms": {"type": "integer", "minimum": 0},
        "evidence_ms": {"type": "integer", "minimum": 0},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "message": {"type": "string"},
        "suggestion": {"type": "string"},
    },
    "required": [
        "code",
        "severity",
        "scope",
        "start_ms",
        "end_ms",
        "evidence_ms",
        "confidence",
        "message",
        "suggestion",
    ],
}


def _audit_response_format(*, image: bool) -> dict[str, Any]:
    properties: dict[str, Any] = {
        "score": {"type": "number", "minimum": 0, "maximum": 100},
        "hard_gate_passed": {"type": "boolean"},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "recommended_usage": {"type": "string"},
        "review_opinion": {"type": "string"},
        "strengths": {"type": "array", "items": {"type": "string"}},
        "improvement_priorities": {
            "type": "array",
            "items": {"type": "string"},
        },
        "issues": {"type": "array", "items": _ISSUE_JSON_SCHEMA},
    }
    required = list(properties)
    if image:
        properties.update(
            {
                "image_scores": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "clarity": {"type": "number", "minimum": 0, "maximum": 100},
                        "visible_rights_safety": {
                            "type": "number",
                            "minimum": 0,
                            "maximum": 100,
                        },
                        "metadata_ai_usability": {
                            "type": "number",
                            "minimum": 0,
                            "maximum": 100,
                        },
                        "duplicate_deduction": {
                            "type": "number",
                            "minimum": 0,
                            "maximum": 40,
                        },
                    },
                    "required": [
                        "clarity",
                        "visible_rights_safety",
                        "metadata_ai_usability",
                        "duplicate_deduction",
                    ],
                },
                "recommended_title": {"type": "string"},
                "image_suitability": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "approved_use_types": {
                            "type": "array",
                            "items": {"type": "string"},
                        }
                    },
                    "required": ["approved_use_types"],
                },
                "rights_assessment": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "risk_level": {
                            "type": "string",
                            "enum": ["LOW", "MEDIUM", "HIGH"],
                        },
                        "reason": {"type": "string"},
                    },
                    "required": ["risk_level", "reason"],
                },
            }
        )
        required.extend(
            [
                "image_scores",
                "recommended_title",
                "image_suitability",
                "rights_assessment",
            ]
        )
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "image_material_audit" if image else "video_material_audit",
            "strict": True,
            "schema": {
                "type": "object",
                "additionalProperties": False,
                "properties": properties,
                "required": required,
            },
        },
    }


def _compact_image_evidence(material: Mapping[str, Any]) -> dict[str, Any]:
    """只给终审模型发送图片分析摘要，明确排除大向量和存储内部字段。"""
    image_analysis = material.get("image_analysis")
    image = image_analysis if isinstance(image_analysis, Mapping) else {}

    def section(name: str, fields: tuple[str, ...]) -> dict[str, Any]:
        value = image.get(name)
        if not isinstance(value, Mapping):
            return {}
        return {key: value[key] for key in fields if key in value}

    return {
        "title": material.get("title", ""),
        "width": material.get("width", 0),
        "height": material.get("height", 0),
        "quality": section(
            "quality",
            (
                "score",
                "width",
                "height",
                "short_edge",
                "resolution_passed",
                "is_blurry",
                "blur_score",
                "exposure_status",
                "brightness",
                "contrast",
                "sharpness",
            ),
        ),
        "layout": section(
            "layout",
            ("alignment_score", "safe_margin_passed", "composition", "subject_position"),
        ),
        "ocr": section("ocr", ("full_text", "regions", "watermark_candidates")),
        "semantic": section(
            "semantic",
            (
                "description",
                "scene",
                "objects",
                "activities",
                "location_candidates",
                "visible_texts",
                "logos",
                "people",
                "privacy_signals",
                "living_public_figure_candidates",
                "company_identity_evidence",
            ),
        ),
        "title_alignment": section(
            "title_alignment",
            ("status", "score", "reason", "claims", "matched_evidence", "unverified_claims"),
        ),
        "rights": section("rights", ("risk_level", "reason")),
        "suitability": section(
            "suitability",
            (
                "article_inline",
                "article_cover",
                "website_hero",
                "thumbnail",
                "marketing_creative",
            ),
        ),
        "duplicate_assessment": material.get("duplicate_assessment", {"matches": []}),
    }


class ElasticsearchAuditStore:
    def __init__(
        self,
        settings: Settings,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._settings = settings
        self._transport = transport

    def _auth(self) -> httpx.BasicAuth | None:
        if not self._settings.media_audit_elasticsearch_username:
            return None
        password = self._settings.media_audit_elasticsearch_password
        return httpx.BasicAuth(
            self._settings.media_audit_elasticsearch_username,
            "" if password is None else password.get_secret_value(),
        )

    def _url(self, alias: str, document_id: str) -> str:
        base = str(self._settings.media_audit_elasticsearch_url).rstrip("/")
        return f"{base}/{alias}/_doc/{document_id}"

    async def get_material(self, asset_id: str) -> dict[str, Any]:
        timeout = httpx.Timeout(self._settings.media_audit_request_timeout_seconds, connect=3)
        async with httpx.AsyncClient(
            timeout=timeout,
            transport=self._transport,
            auth=self._auth(),
            verify=self._settings.media_audit_elasticsearch_verify_tls,
        ) as client:
            response = await client.get(
                self._url(self._settings.media_audit_material_alias, f"asset:{asset_id}")
            )
        if response.status_code == 404:
            raise ProjectionNotReadyError("material projection is not available yet")
        try:
            response.raise_for_status()
            source = response.json()["_source"]
        except (httpx.HTTPError, KeyError, TypeError, ValueError) as exc:
            raise MediaAuditError("failed to read material projection from Elasticsearch") from exc
        if not isinstance(source, dict):
            raise MediaAuditError("material projection has no object source")
        return source

    async def get_result(self, asset_id: str) -> dict[str, Any] | None:
        timeout = httpx.Timeout(self._settings.media_audit_request_timeout_seconds, connect=3)
        async with httpx.AsyncClient(
            timeout=timeout,
            transport=self._transport,
            auth=self._auth(),
            verify=self._settings.media_audit_elasticsearch_verify_tls,
        ) as client:
            response = await client.get(
                self._url(self._settings.media_audit_result_alias, _audit_id(asset_id))
            )
        if response.status_code == 404:
            return None
        try:
            response.raise_for_status()
            source = response.json()["_source"]
        except (httpx.HTTPError, KeyError, TypeError, ValueError) as exc:
            raise MediaAuditError("failed to read cached audit from Elasticsearch") from exc
        return source if isinstance(source, dict) else None

    async def put_result(self, asset_id: str, document: Mapping[str, Any]) -> None:
        timeout = httpx.Timeout(self._settings.media_audit_request_timeout_seconds, connect=3)
        async with httpx.AsyncClient(
            timeout=timeout,
            transport=self._transport,
            auth=self._auth(),
            verify=self._settings.media_audit_elasticsearch_verify_tls,
        ) as client:
            response = await client.put(
                self._url(self._settings.media_audit_result_alias, _audit_id(asset_id)),
                json=dict(document),
            )
        try:
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise MediaAuditError("failed to persist audit result to Elasticsearch") from exc


class JavaAuditCallbackClient:
    def __init__(
        self,
        settings: Settings,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._settings = settings
        self._transport = transport

    async def send(self, event: Mapping[str, Any]) -> None:
        callback_url = self._settings.media_audit_java_callback_url
        secret = self._settings.media_audit_java_callback_secret
        if callback_url is None or secret is None:
            raise MediaAuditError("Java audit callback is not configured")
        body = json.dumps(dict(event), ensure_ascii=False, separators=(",", ":")).encode()
        timestamp = str(int(time.time()))
        signature = hmac.new(
            secret.get_secret_value().encode(),
            timestamp.encode() + b"\n" + body,
            hashlib.sha256,
        ).hexdigest()
        timeout = httpx.Timeout(self._settings.media_audit_callback_timeout_seconds, connect=3)
        async with httpx.AsyncClient(timeout=timeout, transport=self._transport) as client:
            response = await client.post(
                str(callback_url),
                content=body,
                headers={
                    "content-type": "application/json",
                    "x-gta-timestamp": timestamp,
                    "x-gta-signature": f"sha256={signature}",
                },
            )
        try:
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise MediaAuditError("Java audit callback failed") from exc


@dataclass
class _QueuedAudit:
    request: MediaAuditRequest
    sequence: int


class MediaAuditCoordinator:
    """审核编排器：只访问 ES、模型和 Java 的单一签名回调，不含数据库客户端。"""

    def __init__(
        self,
        settings: Settings,
        local_model: LocalModelClient,
        *,
        store: ElasticsearchAuditStore | None = None,
        callback: JavaAuditCallbackClient | None = None,
    ) -> None:
        self._settings = settings
        self._local_model = local_model
        self._store = store or ElasticsearchAuditStore(settings)
        self._callback = callback or JavaAuditCallbackClient(settings)
        self._queue: asyncio.PriorityQueue[tuple[int, int, _QueuedAudit]] = asyncio.PriorityQueue()
        self._states: dict[str, MediaAuditState] = {}
        self._requests: dict[str, MediaAuditRequest] = {}
        self._cancelled: set[str] = set()
        self._sequence = 0
        self._lock = asyncio.Lock()
        self._workers: list[asyncio.Task[None]] = []
        self._retry_tasks: set[asyncio.Task[None]] = set()
        self._transient_attempts: dict[str, int] = {}

    async def start(self) -> None:
        if not self._settings.media_audit_enabled or self._workers:
            return
        _configure_audit_log(self._settings.media_audit_log_path)
        self._workers = [
            asyncio.create_task(self._worker(), name=f"media-audit-{index}")
            for index in range(self._settings.media_audit_max_concurrency)
        ]
        _audit_log(
            "media_audit_runtime_started",
            concurrency=self._settings.media_audit_max_concurrency,
            result_alias=self._settings.media_audit_result_alias,
        )

    async def close(self) -> None:
        workers, self._workers = self._workers, []
        retries, self._retry_tasks = list(self._retry_tasks), set()
        for worker in workers:
            worker.cancel()
        for retry in retries:
            retry.cancel()
        await asyncio.gather(*workers, *retries, return_exceptions=True)

    async def submit(self, request: MediaAuditRequest) -> MediaAuditAccepted:
        async with self._lock:
            existing = self._requests.get(request.request_id)
            state = self._states.get(request.request_id)
            if existing is not None:
                existing_input = existing.model_dump(exclude={"priority"})
                submitted_input = request.model_dump(exclude={"priority"})
                if existing_input != submitted_input:
                    raise MediaAuditError("request_id was reused with different audit input")
                assert state is not None
                if state.state == "failed":
                    # Java 会在历史首轮完成后用同一 request_id 将技术失败降级为
                    # P9 重试；priority 是调度属性，不属于审核幂等输入。
                    self._requests[request.request_id] = request
                    self._transient_attempts.pop(request.request_id, None)
                    self._states[request.request_id] = MediaAuditState(
                        request_id=request.request_id, state="queued"
                    )
                    self._sequence += 1
                    priority = 9 if request.priority == "P9" else 1
                    queued = _QueuedAudit(request=request, sequence=self._sequence)
                    await self._queue.put((priority, self._sequence, queued))
                    return MediaAuditAccepted(
                        request_id=request.request_id, state="queued", idempotent=True
                    )
                accepted_state = (
                    "completed"
                    if state.state == "completed"
                    else ("running" if state.state == "running" else "queued")
                )
                return MediaAuditAccepted(
                    request_id=request.request_id,
                    state=accepted_state,
                    idempotent=True,
                )
            self._requests[request.request_id] = request
            self._states[request.request_id] = MediaAuditState(
                request_id=request.request_id, state="queued"
            )
            self._sequence += 1
            priority = 9 if request.priority == "P9" else 1
            queued = _QueuedAudit(request=request, sequence=self._sequence)
            await self._queue.put((priority, self._sequence, queued))
            _audit_log(
                "media_audit_queued",
                request_id=request.request_id,
                asset_id=request.asset_id,
                media_type=request.media_type,
                priority=request.priority,
                queue_size=self._queue.qsize(),
            )
        return MediaAuditAccepted(request_id=request.request_id, state="queued")

    async def cancel(self, request_id: str) -> MediaAuditState:
        async with self._lock:
            self._cancelled.add(request_id)
            state = MediaAuditState(request_id=request_id, state="cancelled")
            self._states[request_id] = state
            _audit_log("media_audit_cancelled", request_id=request_id)
            return state

    async def status(self, request_id: str) -> MediaAuditState:
        async with self._lock:
            return self._states.get(
                request_id,
                MediaAuditState(request_id=request_id, state="unknown"),
            )

    async def _worker(self) -> None:
        while True:
            _, _, queued = await self._queue.get()
            request = queued.request
            started = time.perf_counter()
            try:
                if await self._is_cancelled(request.request_id):
                    continue
                await self._set_state(request.request_id, "running")
                _audit_log(
                    "media_audit_started",
                    request_id=request.request_id,
                    asset_id=request.asset_id,
                    media_type=request.media_type,
                    priority=request.priority,
                    queue_size=self._queue.qsize(),
                )
                await self._execute(request)
                if not await self._is_cancelled(request.request_id):
                    await self._set_state(request.request_id, "completed")
                    _audit_log(
                        "media_audit_completed",
                        request_id=request.request_id,
                        asset_id=request.asset_id,
                        media_type=request.media_type,
                        duration_ms=round((time.perf_counter() - started) * 1000),
                    )
            except asyncio.CancelledError:
                raise
            except ProjectionNotReadyError as exc:
                await self._set_state(request.request_id, "queued", str(exc))
                retry = asyncio.create_task(
                    self._requeue_after(
                        request, self._settings.media_audit_projection_retry_seconds
                    ),
                    name=f"media-audit-wait-projection-{request.request_id}",
                )
                self._retry_tasks.add(retry)
                retry.add_done_callback(self._retry_tasks.discard)
                _audit_log(
                    "media_audit_waiting_projection",
                    request_id=request.request_id,
                    asset_id=request.asset_id,
                    detail=str(exc),
                    retry_after_seconds=self._settings.media_audit_projection_retry_seconds,
                )
            except Exception as exc:
                attempts = self._transient_attempts.get(request.request_id, 0) + 1
                self._transient_attempts[request.request_id] = attempts
                if attempts <= self._settings.media_audit_transient_retry_max_attempts:
                    await self._set_state(
                        request.request_id,
                        "queued",
                        f"transient retry {attempts}: {exc}",
                    )
                    retry = asyncio.create_task(
                        self._requeue_after(
                            request,
                            self._settings.media_audit_transient_retry_seconds * attempts,
                        ),
                        name=f"media-audit-transient-retry-{request.request_id}-{attempts}",
                    )
                    self._retry_tasks.add(retry)
                    retry.add_done_callback(self._retry_tasks.discard)
                    _audit_log(
                        "media_audit_retry_scheduled",
                        request_id=request.request_id,
                        asset_id=request.asset_id,
                        attempt=attempts,
                        detail=str(exc),
                    )
                else:
                    await self._set_state(request.request_id, "failed", str(exc))
                    _audit_log(
                        "media_audit_failed",
                        level=logging.ERROR,
                        request_id=request.request_id,
                        asset_id=request.asset_id,
                        media_type=request.media_type,
                        attempts=attempts,
                        duration_ms=round((time.perf_counter() - started) * 1000),
                        detail=str(exc),
                    )
                    try:
                        await self._callback.send(self._failure_event(request, exc))
                        _audit_log(
                            "media_audit_failure_callback_completed",
                            request_id=request.request_id,
                            asset_id=request.asset_id,
                        )
                    except Exception as callback_error:
                        _audit_log(
                            "media_audit_failure_callback_failed",
                            level=logging.ERROR,
                            request_id=request.request_id,
                            asset_id=request.asset_id,
                            detail=str(callback_error),
                        )
            finally:
                self._queue.task_done()

    async def _requeue_after(self, request: MediaAuditRequest, delay_seconds: float) -> None:
        await asyncio.sleep(delay_seconds)
        async with self._lock:
            if request.request_id in self._cancelled or not self._workers:
                return
            self._sequence += 1
            priority = 9 if request.priority == "P9" else 1
            queued = _QueuedAudit(request=request, sequence=self._sequence)
            await self._queue.put((priority, self._sequence, queued))

    async def _is_cancelled(self, request_id: str) -> bool:
        async with self._lock:
            return request_id in self._cancelled

    async def _set_state(self, request_id: str, state: str, detail: str | None = None) -> None:
        async with self._lock:
            self._states[request_id] = MediaAuditState(
                request_id=request_id,
                state=state,  # type: ignore[arg-type]
                detail=detail,
            )

    async def _execute(self, request: MediaAuditRequest) -> None:
        stage_started = time.perf_counter()
        material = await self._store.get_material(request.asset_id)
        self._validate_projection(request, material)
        _audit_log(
            "media_audit_projection_loaded",
            request_id=request.request_id,
            asset_id=request.asset_id,
            duration_ms=round((time.perf_counter() - stage_started) * 1000),
        )
        stage_started = time.perf_counter()
        cached = await self._store.get_result(request.asset_id)
        cached_input_hash = (
            cached.get("model_result", {}).get("audit_input_hash")
            if isinstance(cached, dict) and isinstance(cached.get("model_result"), dict)
            else None
        )
        if cached is not None and cached_input_hash == request.audit_input_hash:
            document = dict(cached)
            document["request_id"] = request.request_id
            document["asset_generation"] = request.asset_generation
            document["updated_at"] = _utc_now()
            cache_hit = True
        else:
            document = await self._evaluate(request, material)
            cache_hit = False
        _audit_log(
            "media_audit_result_prepared",
            request_id=request.request_id,
            asset_id=request.asset_id,
            cache_hit=cache_hit,
            decision=document.get("decision"),
            score=document.get("score"),
            duration_ms=round((time.perf_counter() - stage_started) * 1000),
        )
        if await self._is_cancelled(request.request_id):
            return
        stage_started = time.perf_counter()
        await self._store.put_result(request.asset_id, document)
        _audit_log(
            "media_audit_es_written",
            request_id=request.request_id,
            asset_id=request.asset_id,
            duration_ms=round((time.perf_counter() - stage_started) * 1000),
        )
        event = self._completion_event(request, document)
        if await self._is_cancelled(request.request_id):
            return
        stage_started = time.perf_counter()
        await self._callback.send(event)
        _audit_log(
            "media_audit_callback_completed",
            request_id=request.request_id,
            asset_id=request.asset_id,
            duration_ms=round((time.perf_counter() - stage_started) * 1000),
        )

    @staticmethod
    def _validate_projection(request: MediaAuditRequest, material: Mapping[str, Any]) -> None:
        if str(material.get("asset_id", "")) != request.asset_id:
            raise ProjectionNotReadyError("material projection asset_id mismatch")
        if str(material.get("asset_generation", "")) != request.asset_generation:
            raise ProjectionNotReadyError("material projection generation is stale")
        if str(material.get("current_audit_request_id", "")) != request.request_id:
            raise ProjectionNotReadyError("material projection audit request is stale")
        analysis = material.get("analysis")
        if not isinstance(analysis, Mapping) or analysis.get("status") != "READY":
            raise ProjectionNotReadyError("material analysis is not READY")
        if request.media_type == "image":
            image_analysis = material.get("image_analysis")
            if (
                not isinstance(image_analysis, Mapping)
                or str(image_analysis.get("analysis_status", "")).upper() != "READY"
            ):
                raise ProjectionNotReadyError("material image analysis is not READY")
        if str(material.get("deletion_state", "ACTIVE")).upper() != "ACTIVE":
            raise MediaAuditError("material is being deleted")

    async def _evaluate(
        self, request: MediaAuditRequest, material: Mapping[str, Any]
    ) -> dict[str, Any]:
        is_image = request.media_type == "image"
        if is_image:
            image_evidence = _compact_image_evidence(material)
            prompt = (
                "你是绿色旅行网图片素材终审器。worker 已完成 OCR、画质、语义和元信息提取；"
                "你只复核下面的结构化摘要，不重新描述图片，也不得猜测。评分仅由画质完整性60分、"
                "可见水印/侵权/隐私安全40分、重复素材扣分组成。来源、作者、授权字段、ALT、图注、"
                "SEO 文案、元信息是否存在均不得扣分或产生问题。图片名称轻微不准确只给修改建议，"
                "不能拦截；只有名称和画面主体完全不相干才可 BLOCK。发现第三方平台水印、个人联系"
                "方式、明显侵权/隐私风险、令人强烈不适或不可用画质才可 BLOCK。绿色旅行网绿色旗帜"
                "及 GreenTourAsia 标识是我方品牌；多人旅游合影且含该标识属于真实带团优质证据，"
                "不得当作第三方水印或授权缺失。review_opinion 第一段说明画面内容与审核结论。"
                "只返回符合 JSON Schema 的对象。\n图片证据："
                + _bounded_text(image_evidence, 16_000)
            )
            output_tokens = self._settings.media_audit_image_max_output_tokens
        else:
            evidence_fields = (
                "title",
                "media_type",
                "width",
                "height",
                "duration_ms",
                "visual_text",
                "ocr_text",
                "transcript",
                "timeline_ocr",
                "audio_tracks",
                "subtitle_tracks",
                "environment_sounds",
                "video_understanding",
                "quality_evidence",
                "covers",
                "analysis",
            )
            evidence = {key: material[key] for key in evidence_fields if key in material}
            prompt = (
                "你是绿色旅行网视频素材审核器。仅依据证据审核，不得猜测。通过必须同时满足："
                "总分至少80、无BLOCK、画质可用、无第三方水印/联系方式/侵权高风险；"
                "还必须具备前15秒吸引力、我方联系营销话术、业务转化能力。"
                "review_opinion 第一段先说明整个视频讲了什么。"
                "时间轴问题必须指出具体场景和修改方法。"
                "只返回符合 JSON Schema 的对象。\n视频证据：" + _bounded_text(evidence, 60_000)
            )
            output_tokens = self._settings.media_audit_max_output_tokens
        model_started = time.perf_counter()
        response = await self._local_model.generate(
            LocalGenerationRequest(
                messages=[
                    ModelMessage(role="system", content="严格素材审核，只返回JSON。"),
                    ModelMessage(role="user", content=prompt),
                ],
                max_tokens=output_tokens,
                temperature=0.1,
                enable_thinking=False,
                response_format=_audit_response_format(image=is_image),
                priority=request.priority,
            )
        )
        _audit_log(
            "media_audit_model_completed",
            request_id=request.request_id,
            asset_id=request.asset_id,
            media_type=request.media_type,
            duration_ms=round((time.perf_counter() - model_started) * 1000),
            prompt_tokens=response.prompt_tokens,
            completion_tokens=response.completion_tokens,
            total_tokens=response.total_tokens,
            output_chars=len(response.content),
            finish_reason=response.finish_reason,
            model=response.model,
        )
        model_result = _extract_json_object(response.content)
        weighted_score = max(0.0, min(100.0, float(model_result.get("score", 0))))
        issues = self._normalize_issues(model_result.get("issues"), request.media_type)
        analysis = material.get("analysis") if isinstance(material.get("analysis"), Mapping) else {}
        image_fields: dict[str, Any] = {}
        if is_image:
            image_analysis = (
                material.get("image_analysis")
                if isinstance(material.get("image_analysis"), Mapping)
                else analysis
            )
            weighted_score, image_fields = self._normalize_image_scores(
                model_result, issues, image_analysis, weighted_score
            )
        blocking = any(issue["severity"] == "BLOCK" for issue in issues)
        hard_gate = (
            bool(model_result.get("hard_gate_passed", False))
            or (is_image and weighted_score >= 80.0)
        ) and not blocking
        score_cap = 100.0 if hard_gate else 79.0
        score = min(weighted_score, score_cap)
        decision = "PASS" if hard_gate and score >= 80.0 else "REJECT"
        completed_at = _utc_now()
        model_result["audit_input_hash"] = request.audit_input_hash
        document = {
            "schema_version": 3,
            "audit_id": _audit_id(request.asset_id),
            "request_id": request.request_id,
            "asset_id": request.asset_id,
            "asset_generation": request.asset_generation,
            "asset_md5": request.source_md5,
            "source_key": request.source_key,
            "media_type": request.media_type,
            "audit_kind": "IMAGE_CONTENT" if request.media_type == "image" else "VIDEO_CONTENT",
            "decision": decision,
            "score": round(score, 1),
            "weighted_score_before_caps": round(weighted_score, 1),
            "score_cap": score_cap,
            "audit_profile": (
                "image_editorial" if request.media_type == "image" else "direct_conversion"
            ),
            "hard_gate_passed": hard_gate,
            "analysis_complete": True,
            "analysis_version": str(
                (
                    material.get("image_analysis", {})
                    if is_image and isinstance(material.get("image_analysis"), Mapping)
                    else analysis
                ).get("analysis_version", "analysis-v1")
            ),
            "source_etag": str(
                (
                    material.get("image_analysis", {})
                    if is_image and isinstance(material.get("image_analysis"), Mapping)
                    else analysis
                ).get("source_etag", "")
            ),
            "policy_version": request.policy_version,
            "model": response.model,
            "model_confidence": max(0.0, min(1.0, float(model_result.get("confidence", 0.0)))),
            "recommended_usage": str(model_result.get("recommended_usage", "UNUSABLE")),
            "review_opinion": str(model_result.get("review_opinion", "未提供审核意见")),
            "strengths": self._string_list(model_result.get("strengths")),
            "improvement_priorities": self._string_list(model_result.get("improvement_priorities")),
            "gate_results": {
                "objective_quality_passed": hard_gate,
                "blocking_issue_count": sum(1 for issue in issues if issue["severity"] == "BLOCK"),
            },
            "issues": issues,
            "model_result": model_result,
            "created_at": completed_at,
            "updated_at": completed_at,
        }
        document.update(image_fields)
        return document

    @staticmethod
    def _normalize_image_scores(
        model_result: Mapping[str, Any],
        issues: list[dict[str, Any]],
        analysis: Mapping[str, Any],
        model_score: float,
    ) -> tuple[float, dict[str, Any]]:
        raw_scores = model_result.get("image_scores")
        scores = raw_scores if isinstance(raw_scores, Mapping) else {}

        def bounded_number(value: object, fallback: float) -> float:
            try:
                return max(0.0, min(100.0, float(value)))
            except (TypeError, ValueError):
                return fallback

        rights_pattern = re.compile(
            r"watermark|hiddenmark|copyright|infringement|qrcode|contact|privacy|"
            r"publicfigure|celebrity|水印|暗记|侵权|二维码|联系方式|隐私|肖像",
            re.IGNORECASE,
        )
        rights_issues = [
            issue
            for issue in issues
            if rights_pattern.search(f"{issue.get('code', '')} {issue.get('message', '')}")
        ]
        derived_rights = 100.0
        if any(issue.get("severity") == "BLOCK" for issue in rights_issues):
            derived_rights = 0.0
        elif any(issue.get("severity") == "ERROR" for issue in rights_issues):
            derived_rights = 40.0
        elif rights_issues:
            derived_rights = 70.0
        rights = min(
            bounded_number(scores.get("visible_rights_safety"), derived_rights),
            derived_rights,
        )
        duplicate_deduction = max(
            0.0, min(40.0, bounded_number(scores.get("duplicate_deduction"), 0.0))
        )

        quality = analysis.get("quality") if isinstance(analysis.get("quality"), Mapping) else {}
        objective_quality = quality.get("score")
        explicit_clarity = scores.get("clarity", scores.get("technical_quality"))
        if objective_quality is not None:
            # 清晰度必须以 worker 对原图的客观测量为准，避免终审模型生成一个
            # 与总分不相干的 0 分或 100 分。
            clarity = bounded_number(objective_quality, 0.0)
        elif explicit_clarity is None:
            # 兼容旧模型：反解画质分，使两个分项与模型总分严格相加，绝不
            # 再出现总分88而画质显示0的假数据。
            clarity = max(
                0.0,
                min(100.0, (model_score + duplicate_deduction - rights * 0.4) / 0.6),
            )
        else:
            clarity = bounded_number(explicit_clarity, 0.0)

        metadata_ready = (
            100.0
            if str(analysis.get("analysis_status", analysis.get("status", ""))).upper() == "READY"
            else 0.0
        )
        # 元信息是系统内部的 AI 可用性标记，只展示事实，不参与总分。
        metadata_usability = metadata_ready
        clarity_contribution = round(clarity * 0.6, 1)
        rights_contribution = round(rights * 0.4, 1)
        normalized_total = max(
            0.0,
            min(100.0, clarity_contribution + rights_contribution - duplicate_deduction),
        )

        raw_rights = model_result.get("rights_assessment")
        rights_assessment = dict(raw_rights) if isinstance(raw_rights, Mapping) else {}
        rights_assessment.setdefault(
            "risk_level", "HIGH" if rights < 50 else "MEDIUM" if rights < 80 else "LOW"
        )
        rights_assessment.setdefault(
            "reason",
            (
                "发现可见水印、侵权或隐私风险"
                if rights_issues
                else "未发现可见水印、第三方平台暗记或侵权高风险"
            ),
        )
        raw_suitability = model_result.get("image_suitability")
        image_suitability = dict(raw_suitability) if isinstance(raw_suitability, Mapping) else {}
        approved = image_suitability.get("approved_use_types")
        if not isinstance(approved, list):
            image_suitability["approved_use_types"] = []

        raw_duplicates = model_result.get("duplicate_assessment")
        duplicate_assessment = (
            dict(raw_duplicates) if isinstance(raw_duplicates, Mapping) else {"matches": []}
        )
        if not isinstance(duplicate_assessment.get("matches"), list):
            duplicate_assessment["matches"] = []

        return round(normalized_total, 1), {
            "image_scores": {
                "clarity": round(clarity, 1),
                "technical_quality": round(clarity, 1),
                "clarity_contribution": clarity_contribution,
                "visible_rights_safety": round(rights, 1),
                "visible_rights_contribution": rights_contribution,
                "metadata_ai_usability": round(metadata_usability, 1),
                "duplicate_deduction": round(duplicate_deduction, 1),
            },
            "image_suitability": image_suitability,
            "rights_assessment": rights_assessment,
            "duplicate_assessment": duplicate_assessment,
            "recommended_title": str(model_result.get("recommended_title", ""))[:64],
        }

    @staticmethod
    def _string_list(value: object) -> list[str]:
        if not isinstance(value, list):
            return []
        return [str(item)[:2000] for item in value if str(item).strip()]

    @staticmethod
    def _normalize_issues(value: object, media_type: str) -> list[dict[str, Any]]:
        if not isinstance(value, list):
            return []
        result: list[dict[str, Any]] = []
        for index, raw in enumerate(value):
            if not isinstance(raw, Mapping):
                continue
            code = str(raw.get("code", "UNCLASSIFIED"))[:128]
            message = str(raw.get("message", "审核问题"))[:4000]
            suggestion = str(raw.get("suggestion", "请根据证据修改"))[:4000]
            if media_type == "image":
                source_only = re.search(
                    r"source.?missing|author.?missing|license.?missing|metadata.?missing|"
                    r"alt.?missing|seo|可信来源|来源无法验证|无法验证来源|作者缺失|"
                    r"授权字段缺失|元信息缺失|补充来源|补写alt|补写图注",
                    f"{code} {message} {suggestion}",
                    re.IGNORECASE,
                )
                visible_risk = re.search(
                    r"watermark|hiddenmark|infringement|privacy|水印|暗记|侵权|隐私|"
                    r"二维码|联系方式",
                    f"{code} {message}",
                    re.IGNORECASE,
                )
                # 这些信息由系统内部读取/清洗，既不能让用户处理，也不属于图片
                # 画质与可见侵权审核。模型偶发生成时在服务端硬过滤。
                if source_only and not visible_risk:
                    continue
            severity = str(raw.get("severity", "WARN")).upper()
            if severity not in {"BLOCK", "ERROR", "WARN"}:
                severity = "WARN"
            if (
                media_type == "image"
                and severity == "BLOCK"
                and re.search(r"title|name|名称|标题", code, re.IGNORECASE)
                and not re.search(r"完全不|毫不相干|无关|不搭边", message)
            ):
                severity = "WARN"
            scope = str(raw.get("scope", "IMAGE" if media_type == "image" else "TIMELINE")).upper()
            if scope not in {"GLOBAL", "TIMELINE", "IMAGE", "DELIVERY_CONFIG"}:
                scope = "IMAGE" if media_type == "image" else "TIMELINE"
            start_ms = max(0, int(raw.get("start_ms", 0) or 0))
            end_ms = max(start_ms, int(raw.get("end_ms", start_ms) or start_ms))
            evidence_ms = max(0, int(raw.get("evidence_ms", start_ms) or start_ms))
            result.append(
                {
                    "issue_id": f"issue-{index + 1}",
                    "code": code,
                    "severity": severity,
                    "scope": scope,
                    "evidence_kind": "image" if media_type == "image" else "timeline",
                    "start_ms": start_ms,
                    "end_ms": end_ms,
                    "evidence_ms": evidence_ms,
                    "confidence": max(0.0, min(1.0, float(raw.get("confidence", 0.0) or 0.0))),
                    "message": message,
                    "suggestion": suggestion,
                }
            )
        return result

    @staticmethod
    def _failure_event(request: MediaAuditRequest, error: Exception) -> dict[str, Any]:
        failed_at = _utc_now()
        return {
            "schema_version": 1,
            "event_type": "material.ai.audit.failed",
            "event_id": _event_id(request.request_id, failed_at),
            "request_id": request.request_id,
            "asset_id": request.asset_id,
            "asset_generation": request.asset_generation,
            "source_key": request.source_key,
            "source_md5": request.source_md5,
            "policy_version": request.policy_version,
            "failure_code": type(error).__name__,
            "failure_reason": str(error)[:512],
            "failed_at": failed_at,
        }

    @staticmethod
    def _completion_event(
        request: MediaAuditRequest, document: Mapping[str, Any]
    ) -> dict[str, Any]:
        completed_at = _utc_now()
        event = {
            "schema_version": 3,
            "event_type": "material.ai.audit.completed",
            "event_id": _event_id(request.request_id, completed_at),
            "audit_id": str(document["audit_id"]),
            "request_id": request.request_id,
            "asset_id": request.asset_id,
            "asset_generation": request.asset_generation,
            "asset_md5": request.source_md5,
            "source_key": request.source_key,
            "media_type": request.media_type,
            "audit_kind": str(document["audit_kind"]),
            "analysis_version": str(document["analysis_version"]),
            "source_etag": str(document.get("source_etag", "")),
            "policy_version": request.policy_version,
            "decision": str(document["decision"]),
            "audit_profile": str(document["audit_profile"]),
            "score": float(document["score"]),
            "hard_gate_passed": bool(document["hard_gate_passed"]),
            "model_confidence": float(document.get("model_confidence", 0.0)),
            "weighted_score_before_caps": float(
                document.get("weighted_score_before_caps", document["score"])
            ),
            "score_cap": float(document.get("score_cap", 100.0)),
            "recommended_usage": str(document.get("recommended_usage", "UNUSABLE")),
            "review_opinion": str(document.get("review_opinion", "")),
            "strengths": list(document.get("strengths", [])),
            "improvement_priorities": list(document.get("improvement_priorities", [])),
            "issues": list(document.get("issues", [])),
            "completed_at": completed_at,
        }
        # 前端读取的是 Java 事务保存的完成事件，而不是直接读取 ES 审核文档。
        # 分项必须随事件传递，不能让前端因字段缺失把它伪装成 0 分。
        for field in (
            "analysis_complete",
            "image_scores",
            "image_suitability",
            "rights_assessment",
            "duplicate_assessment",
            "recommended_title",
            "score_breakdown",
            "dimension_scores",
            "cover_scores",
        ):
            if field in document:
                event[field] = document[field]
        return event


class EmbeddedMaterialAuditEvaluator(MediaAuditCoordinator):
    """只执行模型推理；素材证据由 projection 随 Kafka 任务完整传入。"""

    def __init__(self, settings: Settings, local_model: LocalModelClient) -> None:
        # 故意不调用旧 HTTP 编排器构造函数，避免创建 ES store、Java callback
        # 或进程内任务队列。运行时只保留提示词、规则归一化和模型客户端。
        self._settings = settings
        self._local_model = local_model

    async def evaluate_embedded(
        self, request: MediaAuditRequest, material: Mapping[str, Any]
    ) -> dict[str, Any]:
        self._validate_embedded(request, material)
        return await self._evaluate(request, material)

    @staticmethod
    def _validate_embedded(
        request: MediaAuditRequest, material: Mapping[str, Any]
    ) -> None:
        if str(material.get("asset_id", "")) != request.asset_id:
            raise MediaAuditError("embedded material asset_id mismatch")
        if str(material.get("asset_generation", "")) != request.asset_generation:
            raise MediaAuditError("embedded material generation mismatch")
        if str(material.get("deletion_state", "ACTIVE")).upper() != "ACTIVE":
            raise MediaAuditError("material is being deleted")
        analysis = material.get("analysis")
        if not isinstance(analysis, Mapping) or analysis.get("status") != "READY":
            raise MediaAuditError("embedded material analysis is not READY")
        if request.media_type == "image":
            image_analysis = material.get("image_analysis")
            if (
                not isinstance(image_analysis, Mapping)
                or str(image_analysis.get("analysis_status", "")).upper() != "READY"
            ):
                raise MediaAuditError("embedded image analysis is not READY")

    @staticmethod
    def completion_event(
        request: MediaAuditRequest, document: Mapping[str, Any]
    ) -> dict[str, Any]:
        event = MediaAuditCoordinator._completion_event(request, document)
        # gta-ai 不再写 ES；projection 消费这个完整文档后负责 V1 覆盖写入。
        event["audit_document"] = dict(document)
        return event

    @staticmethod
    def failure_event(request: MediaAuditRequest, error: Exception) -> dict[str, Any]:
        return MediaAuditCoordinator._failure_event(request, error)
