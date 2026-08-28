from __future__ import annotations

import atexit
import copy
import json
import logging
import logging.handlers
import os
import queue
import socket
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


class _JsonFormatter(logging.Formatter):
    """文件和Kafka共用同一种简单、可检索的JSON格式。"""

    def __init__(self, service: str) -> None:
        super().__init__()
        self.service = service
        self.hostname = socket.gethostname() or "unknown"

    def format(self, record: logging.LogRecord) -> str:
        value: dict[str, Any] = {
            "time": datetime.fromtimestamp(record.created, UTC).isoformat(
                timespec="milliseconds"
            ).replace("+00:00", "Z"),
            "level": record.levelname,
            "hostname": self.hostname,
            "service": self.service,
            "message": record.getMessage(),
        }
        value.update(getattr(record, "fields", {}))
        if record.exc_info:
            value["stacktrace"] = self.formatException(record.exc_info)
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=str)


class _ConsoleFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return record.levelno >= logging.ERROR or bool(getattr(record, "console", False))


class _ConsoleFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        now = (
            datetime.fromtimestamp(record.created, UTC).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3]
            + "Z"
        )
        line = f"[{now}][{record.levelname}][{socket.gethostname()}] {record.getMessage()}"
        if record.getMessage().startswith("27B心跳") and sys.stderr.isatty():
            return f"\033[94m{line}\033[0m"
        return line


class _QueueHandler(logging.handlers.QueueHandler):
    """普通日志不阻塞推理; 错误日志最多等待一秒, 尽量避免丢失。"""

    def prepare(self, record: logging.LogRecord) -> logging.LogRecord:
        prepared = copy.copy(record)
        prepared.msg = record.getMessage()
        prepared.args = None
        return prepared

    def enqueue(self, record: logging.LogRecord) -> None:
        try:
            self.queue.put_nowait(record)
        except queue.Full:
            if record.levelno >= logging.ERROR:
                self.queue.put(record, timeout=1)


class _KafkaHandler(logging.Handler):
    def __init__(self, brokers: str, topic: str, service: str) -> None:
        super().__init__()
        from confluent_kafka import Producer

        self.producer = Producer(
            {
                "bootstrap.servers": brokers,
                "client.id": f"{service}-logger",
                "acks": "all",
                "enable.idempotence": True,
                "log_level": 0,
            }
        )
        self.topic = topic
        self.key = f"{service}:{socket.gethostname()}".encode()

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self.producer.produce(
                self.topic,
                key=self.key,
                value=self.format(record).encode(),
            )
            self.producer.poll(0)
        except Exception:
            # Kafka异常不能影响27B推理, 本地滚动文件仍会保留同一条日志。
            pass

    def close(self) -> None:
        self.producer.flush(2)
        super().close()


_logger = logging.getLogger("gta_ai_27b")
_listener: logging.handlers.QueueListener | None = None


def configure(service: str) -> None:
    """整个27B链路只调用一次, 控制台、文件和Kafka自动分流。"""
    global _listener
    if _listener is not None:
        return

    log_dir = Path(os.getenv("GTA_AI_LOG_DIR", "/opt/gta-ai/qwen-llm/logs"))
    log_dir.mkdir(parents=True, exist_ok=True)
    max_bytes = int(os.getenv("GTA_AI_LOG_MAX_MB", "20")) * 1024 * 1024
    backups = int(os.getenv("GTA_AI_LOG_BACKUPS", "10"))
    formatter = _JsonFormatter(service)
    runtime_name, error_name = (
        ("router.log", "error.log")
        if service == "gta-ai-router"
        else ("vllm.log", "vllm.error.log")
    )

    console = logging.StreamHandler()
    console.addFilter(_ConsoleFilter())
    console.setFormatter(_ConsoleFormatter())

    runtime_file = logging.handlers.RotatingFileHandler(
        log_dir / runtime_name, maxBytes=max_bytes, backupCount=backups, encoding="utf-8"
    )
    runtime_file.setFormatter(formatter)

    error_file = logging.handlers.RotatingFileHandler(
        log_dir / error_name, maxBytes=max_bytes, backupCount=backups, encoding="utf-8"
    )
    error_file.setLevel(logging.ERROR)
    error_file.setFormatter(formatter)

    handlers: list[logging.Handler] = [console, runtime_file, error_file]
    if os.getenv("GTA_AI_LOG_KAFKA_ENABLED", "true").lower() in {"1", "true", "yes"}:
        try:
            kafka = _KafkaHandler(
                os.getenv("GTA_AI_LOG_KAFKA_BROKERS", ""),
                os.getenv("GTA_AI_LOG_KAFKA_TOPIC", "gta.video.worker.logs"),
                service,
            )
            kafka.setFormatter(formatter)
            handlers.append(kafka)
        except Exception as exc:
            print(f"[WARN] Kafka日志不可用, 已降级为本地文件: {exc}", file=sys.stderr)

    log_queue: queue.Queue[logging.LogRecord] = queue.Queue(maxsize=8192)
    _logger.handlers.clear()
    _logger.addHandler(_QueueHandler(log_queue))
    _logger.setLevel(logging.INFO)
    _logger.propagate = False
    _listener = logging.handlers.QueueListener(log_queue, *handlers, respect_handler_level=True)
    _listener.start()
    atexit.register(shutdown)


def info(message: str, *, console: bool = False, **fields: Any) -> None:
    _logger.info(message, extra={"console": console, "fields": fields})


def error(message: str, *, exc_info: bool = False, **fields: Any) -> None:
    _logger.error(message, extra={"console": True, "fields": fields}, exc_info=exc_info)


def shutdown() -> None:
    global _listener
    if _listener is not None:
        _listener.stop()
        _listener = None
