from __future__ import annotations

import asyncio
import json
import logging
import os
import socket
import time
from collections.abc import Mapping
from typing import Any, Protocol

from gta_ai.clients.local_model import LocalModelClient
from gta_ai.config import Settings
from gta_ai.media_audit import (
    EmbeddedMaterialAuditEvaluator,
    JavaAuditCallbackClient,
    MediaAuditError,
    _audit_log,
    _configure_audit_log,
)
from gta_ai.schemas import MediaAuditRequest


class MediaAuditWorkerPort(Protocol):
    async def start(self) -> None: ...

    async def close(self) -> None: ...


class KafkaMediaAuditWorker:
    """Kafka 驱动的纯推理执行器, 不发现任务、不扫描数据, 也不读写 ES。"""

    def __init__(self, settings: Settings, local_model: LocalModelClient) -> None:
        self._settings = settings
        self._evaluator = EmbeddedMaterialAuditEvaluator(settings, local_model)
        self._callback = JavaAuditCallbackClient(settings)
        self._consumers: list[Any] = []
        self._tasks: list[asyncio.Task[None]] = []
        self._active: dict[tuple[str, str], asyncio.Task[None]] = {}
        self._active_lock = asyncio.Lock()
        self._result_cache: dict[str, dict[str, Any]] = {}
        self._stopping = False
        self._completed = 0
        self._failed = 0
        self._average_seconds: float | None = None

    async def start(self) -> None:
        if not self._settings.media_audit_enabled or self._tasks:
            return
        try:
            from confluent_kafka import Consumer, Producer
        except ImportError as exc:  # pragma: no cover - deployment dependency check
            raise RuntimeError("confluent-kafka is required for media audit") from exc
        _configure_audit_log(self._settings.media_audit_log_path)
        self._stopping = False
        self._producer = Producer(
            {"bootstrap.servers": self._settings.media_audit_kafka_bootstrap_servers}
        )
        for index in range(self._settings.media_audit_max_concurrency):
            consumer = Consumer(
                {
                    "bootstrap.servers": self._settings.media_audit_kafka_bootstrap_servers,
                    "group.id": self._settings.media_audit_kafka_group_id,
                    "auto.offset.reset": self._settings.media_audit_kafka_auto_offset_reset,
                    "enable.auto.commit": False,
                    "enable.auto.offset.store": False,
                    "max.poll.interval.ms": 3_600_000,
                }
            )
            consumer.subscribe([self._settings.media_audit_kafka_request_topic])
            self._consumers.append(consumer)
            self._tasks.append(
                asyncio.create_task(self._consume(consumer, index), name=f"audit-kafka-{index}")
            )
        cancel_consumer = Consumer(
            {
                "bootstrap.servers": self._settings.media_audit_kafka_bootstrap_servers,
                "group.id": (
                    f"{self._settings.media_audit_kafka_group_id}-cancel-"
                    f"{socket.gethostname()}-{os.getpid()}"
                ),
                "auto.offset.reset": "latest",
                "enable.auto.commit": True,
            }
        )
        cancel_consumer.subscribe([self._settings.media_audit_kafka_cancel_topic])
        self._consumers.append(cancel_consumer)
        self._tasks.append(
            asyncio.create_task(self._consume_cancellations(cancel_consumer), name="audit-cancel")
        )
        _audit_log(
            "media_audit_kafka_started",
            topic=self._settings.media_audit_kafka_request_topic,
            result_topic=self._settings.media_audit_kafka_result_topic,
            concurrency=len(self._tasks),
        )

    async def close(self) -> None:
        self._stopping = True
        tasks, self._tasks = self._tasks, []
        for task in tasks:
            task.cancel()
        async with self._active_lock:
            for task in self._active.values():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        consumers, self._consumers = self._consumers, []
        for consumer in consumers:
            await asyncio.to_thread(consumer.close)
        producer = getattr(self, "_producer", None)
        if producer is not None:
            await asyncio.to_thread(producer.flush, 10)

    async def _consume(self, consumer: Any, worker_index: int) -> None:
        while not self._stopping:
            message = await asyncio.to_thread(
                consumer.poll, self._settings.media_audit_kafka_poll_timeout_seconds
            )
            if message is None:
                continue
            if message.error():
                _audit_log(
                    "media_audit_kafka_poll_failed",
                    level=logging.ERROR,
                    worker_index=worker_index,
                    detail=str(message.error()),
                )
                continue
            try:
                event = json.loads(message.value().decode("utf-8"))
                if not isinstance(event, dict):
                    raise MediaAuditError("audit task must be a JSON object")
                asset_key = (
                    str(event.get("asset_id", "")),
                    str(event.get("asset_generation", "")),
                )
                evaluation = asyncio.create_task(self._process(event))
                async with self._active_lock:
                    self._active[asset_key] = evaluation
                try:
                    await evaluation
                except asyncio.CancelledError:
                    if self._stopping:
                        raise
                    _audit_log(
                        "media_audit_item_cancelled_by_p0",
                        asset_id=asset_key[0],
                        asset_generation=asset_key[1],
                    )
                finally:
                    async with self._active_lock:
                        if self._active.get(asset_key) is evaluation:
                            self._active.pop(asset_key, None)
                await asyncio.to_thread(consumer.commit, message=message, asynchronous=False)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                # 未能把完成/失败结果交给 Kafka broker 时绝不能提交源 offset。
                _audit_log(
                    "media_audit_message_not_committed",
                    level=logging.ERROR,
                    worker_index=worker_index,
                    detail=str(exc),
                )
                await asyncio.sleep(5)

    async def _consume_cancellations(self, consumer: Any) -> None:
        while not self._stopping:
            message = await asyncio.to_thread(
                consumer.poll, self._settings.media_audit_kafka_poll_timeout_seconds
            )
            if message is None or message.error():
                continue
            try:
                event = json.loads(message.value().decode("utf-8"))
                key = (
                    str(event.get("asset_id", "")),
                    str(event.get("asset_generation", "")),
                )
                async with self._active_lock:
                    active = self._active.get(key)
                    if active is not None:
                        active.cancel()
                _audit_log(
                    "media_audit_cancel_observed",
                    asset_id=key[0],
                    asset_generation=key[1],
                    request_id=str(event.get("request_id", "")),
                    active_cancelled=active is not None,
                )
            except Exception as exc:
                _audit_log(
                    "media_audit_cancel_invalid",
                    level=logging.ERROR,
                    detail=str(exc),
                )

    async def _process(self, event: Mapping[str, Any]) -> None:
        material = event.get("material")
        if not isinstance(material, Mapping):
            raise MediaAuditError("projection task has no embedded material evidence")
        requested_at = event.get("requested_at")
        if not requested_at:
            from datetime import UTC, datetime

            requested_at = datetime.now(UTC).isoformat()
        request = MediaAuditRequest.model_validate(
            {
                "schema_version": 1,
                "request_id": event.get("request_id"),
                "asset_id": event.get("asset_id"),
                "asset_generation": event.get("asset_generation"),
                "media_type": event.get("media_type"),
                "source_key": event.get("source_key"),
                "source_md5": event.get("source_md5"),
                "audit_input_hash": event.get("audit_input_hash"),
                "policy_version": event.get("policy_version"),
                "priority": event.get("priority", "P9"),
                "requested_at": requested_at,
            }
        )
        progress = event.get("progress") if isinstance(event.get("progress"), Mapping) else {}
        started = time.perf_counter()
        _audit_log(
            "media_audit_item_started",
            ordinal=progress.get("ordinal", 0),
            total=progress.get("total", 0),
            material_name=str(event.get("material_name", "")),
            asset_id=request.asset_id,
            media_type=request.media_type,
            completed=self._completed,
            failed=self._failed,
            remaining=progress.get("remaining", 0),
            estimated_item_seconds=round(self._average_seconds or 0, 1),
            estimated_total_seconds=round(
                (self._average_seconds or 0) * int(progress.get("remaining", 0) or 0), 1
            ),
        )
        try:
            document = self._result_cache.get(request.audit_input_hash)
            if document is None:
                document = await self._evaluator.evaluate_embedded(request, material)
                if len(self._result_cache) >= 1000:
                    self._result_cache.pop(next(iter(self._result_cache)))
                self._result_cache[request.audit_input_hash] = document
        except Exception as exc:
            result = self._evaluator.failure_event(request, exc)
            await self._publish(result, request.asset_id)
            self._failed += 1
            outcome = "failed"
        else:
            result = self._evaluator.completion_event(request, document)
            await self._publish(result, request.asset_id)
            # 这是唯一允许的业务回写入口; Java 只在事务内应用最终审核结果。
            # 回调失败时抛出并保留源 offset; 重投将命中上面的输入哈希缓存。
            await self._callback.send(result)
            self._completed += 1
            outcome = "completed"
        elapsed = time.perf_counter() - started
        self._average_seconds = (
            elapsed
            if self._average_seconds is None
            else self._average_seconds * 0.8 + elapsed * 0.2
        )
        _audit_log(
            "media_audit_item_finished",
            outcome=outcome,
            asset_id=request.asset_id,
            material_name=str(event.get("material_name", "")),
            duration_seconds=round(elapsed, 1),
            completed=self._completed,
            failed=self._failed,
            remaining=progress.get("remaining", 0),
            estimated_total_seconds=round(
                (self._average_seconds or 0) * int(progress.get("remaining", 0) or 0), 1
            ),
        )

    async def _publish(self, event: Mapping[str, Any], key: str) -> None:
        payload = json.dumps(dict(event), ensure_ascii=False, separators=(",", ":")).encode()

        def send() -> None:
            errors: list[str] = []

            def delivered(error: object, _: object) -> None:
                if error is not None:
                    errors.append(str(error))

            self._producer.produce(
                self._settings.media_audit_kafka_result_topic,
                key=key.encode(),
                value=payload,
                on_delivery=delivered,
            )
            outstanding = self._producer.flush(30)
            if outstanding or errors:
                raise MediaAuditError(
                    "audit result was not acknowledged by Kafka: "
                    + (errors[0] if errors else f"{outstanding} outstanding")
                )

        await asyncio.to_thread(send)
