"""JSON行应用日志（stdout），与 GTA 统一日志字段一致。

本文件在每个模型项目内各保留一份（项目之间禁止相互导入代码）。只依赖标准库。

- 每行一个 JSON：@timestamp、env、service、project、node、host、deploy_sha、level、logger、event、
  message、trace_id、request_id、task_id、duration_ms、result、error、fields。
- env/node/deploy_sha 来自部署注入的 GTA_ENV/GTA_NODE/GTA_DEPLOY_SHA，缺失写 unknown。
- 日志进入有界队列由后台线程写 stdout，队列满时丢弃并计数，不阻塞推理线程。
- InferenceLogMiddleware：每个请求一条完成事件（推理接口 event=model_infer），
  4xx WARN，5xx/异常 ERROR+堆栈。
- 心跳：启动即发一条 event=log_heartbeat，之后每 300 秒一条。
"""

from __future__ import annotations

import copy
import json
import logging
import logging.handlers
import os
import queue
import re
import socket
import sys
import threading
import time
from contextlib import suppress
from contextvars import ContextVar
from datetime import UTC, datetime
from typing import Any

PROJECT = "gta-ai"
HEARTBEAT_INTERVAL_SECONDS = 300
QUEUE_CAPACITY = 10_000
FIELD_MAX_CHARS = 300
_ID_PATTERN = re.compile(r"^[A-Za-z0-9._:@/-]{1,160}$")
_LEVEL_NAMES = {"WARNING": "WARN", "CRITICAL": "FATAL"}
_REDACTED = "[REDACTED]"
_SECRET_ASSIGNMENT = re.compile(
    r"(?i)(password|passwd|secret|api[_-]?key|token|authorization)"
    r"(\s*[\"']?\s*[=:]\s*[\"']?)((?:(?:bearer|basic)\s+)?[^\s\"',;&]+)"
)
_BEARER = re.compile(r"(?i)\b(bearer|basic)\s+[A-Za-z0-9._~+/=-]{8,}")
_SK_KEY = re.compile(r"\bsk-[A-Za-z0-9_-]{8,}")
_URL_QUERY = re.compile(r"(\bhttps?://[^\s?#\"'<>]+)\?([^\s#\"'<>]*)")
_ASGI_DUPLICATE_MESSAGE = "Exception in ASGI application"
# 任务ID来源优先级：Worker 应传 X-GTA-Task-ID，其次通用请求ID。
_TASK_HEADERS = (b"x-gta-task-id", b"x-request-id", b"x-gta-request-id")

_context: ContextVar[dict[str, Any] | None] = ContextVar("gta_ai_log_context", default=None)
_started = time.monotonic()
_dropped_lock = threading.Lock()
_dropped_total = 0
_handler: _BoundedQueueHandler | None = None
_heartbeat_thread: threading.Thread | None = None
_service = "unknown"


def redact(text: str) -> str:
    if not text:
        return text
    text = _SECRET_ASSIGNMENT.sub(lambda m: f"{m.group(1)}{m.group(2)}{_REDACTED}", text)
    text = _BEARER.sub(lambda m: f"{m.group(1)} {_REDACTED}", text)
    text = _SK_KEY.sub(_REDACTED, text)
    return _URL_QUERY.sub(_mask_query, text)


def _mask_query(match: re.Match[str]) -> str:
    pairs = [
        f"{pair.split('=', 1)[0]}=***" if "=" in pair else pair
        for pair in match.group(2).split("&") if pair
    ]
    return f"{match.group(1)}?{'&'.join(pairs)}"


def _clean_fields(fields: dict[str, Any]) -> dict[str, Any]:
    cleaned: dict[str, Any] = {}
    for key, value in list(fields.items())[:30]:
        if value is None:
            continue
        if isinstance(value, str):
            value = redact(value)
            if len(value) > FIELD_MAX_CHARS:
                value = value[:FIELD_MAX_CHARS] + "…"
        elif not isinstance(value, (int, float, bool)):
            value = redact(str(value))[:FIELD_MAX_CHARS]
        cleaned[key] = value
    return cleaned


def _env(name: str) -> str:
    return os.environ.get(name, "").strip() or "unknown"


class JsonFormatter(logging.Formatter):
    def __init__(self, service: str) -> None:
        super().__init__()
        self._service = service
        self._env = _env("GTA_ENV")
        self._node = _env("GTA_NODE")
        self._sha = _env("GTA_DEPLOY_SHA")
        self._host = socket.gethostname()

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "@timestamp": datetime.fromtimestamp(record.created, tz=UTC)
            .isoformat(timespec="milliseconds").replace("+00:00", "Z"),
            "env": self._env,
            "service": self._service,
            "project": PROJECT,
            "node": self._node,
            "host": self._host,
            "deploy_sha": self._sha,
            "level": _LEVEL_NAMES.get(record.levelname, record.levelname),
            "logger": record.name,
            "event": getattr(record, "event", None),
            "message": redact(record.getMessage()),
        }
        for key in ("trace_id", "request_id", "task_id", "result"):
            value = getattr(record, key, None)
            if value:
                payload[key] = value
        duration_ms = getattr(record, "duration_ms", None)
        if duration_ms is not None:
            payload["duration_ms"] = int(duration_ms)
        if record.exc_info and not record.exc_text:
            record.exc_text = self.formatException(record.exc_info)
        if record.exc_text:
            payload["error"] = redact(record.exc_text)
        fields = getattr(record, "fields", None)
        if isinstance(fields, dict) and fields:
            payload["fields"] = _clean_fields(fields)
        return json.dumps(payload, ensure_ascii=False, default=str)


class _ContextFilter(logging.Filter):
    """在产生日志的线程中把请求关联ID写入记录（后台写出线程没有请求上下文）。"""

    def filter(self, record: logging.LogRecord) -> bool:
        context = _context.get()
        if context:
            for key in ("trace_id", "request_id", "task_id"):
                if getattr(record, key, None) is None and context.get(key):
                    setattr(record, key, context[key])
        return True


class DuplicateAsgiErrorFilter(logging.Filter):
    """中间件已记录 ERROR+堆栈后，uvicorn 还会再记一条无上下文的同一异常：降为 DEBUG。"""

    def filter(self, record: logging.LogRecord) -> bool:
        if str(record.msg).startswith(_ASGI_DUPLICATE_MESSAGE) and record.levelno > logging.DEBUG:
            record.levelno, record.levelname = logging.DEBUG, "DEBUG"
            return logging.getLogger().isEnabledFor(logging.DEBUG)
        return True


class _Listener(logging.handlers.QueueListener):
    def enqueue_sentinel(self) -> None:
        with suppress(queue.Full):
            self.queue.put(self._sentinel, timeout=5)


class _BoundedQueueHandler(logging.handlers.QueueHandler):
    def __init__(self, target: logging.Handler, capacity: int = QUEUE_CAPACITY) -> None:
        super().__init__(queue.Queue(maxsize=capacity))
        self.target = target
        self.listener = _Listener(self.queue, target, respect_handler_level=True)
        self.listener.start()
        self._stopped = False

    def prepare(self, record: logging.LogRecord) -> logging.LogRecord:
        prepared = copy.copy(record)
        try:
            message = record.getMessage()
        except Exception:  # 格式化失败不能让日志崩溃
            message = str(record.msg)
        prepared.msg, prepared.args, prepared.message = message, None, message
        if record.exc_info:
            prepared.exc_text = prepared.exc_text or logging.Formatter().formatException(
                record.exc_info)
            prepared.exc_info = None
        return prepared

    def enqueue(self, record: logging.LogRecord) -> None:
        global _dropped_total
        try:
            self.queue.put_nowait(record)
        except queue.Full:
            with _dropped_lock:
                _dropped_total += 1

    def flush(self) -> None:
        deadline = time.monotonic() + 5
        while not self._stopped and self.queue.unfinished_tasks and time.monotonic() < deadline:
            time.sleep(0.005)
        self.target.flush()

    def close(self) -> None:
        if not self._stopped:
            self._stopped = True
            self.listener.stop()
        super().close()


def queue_stats() -> dict[str, int]:
    handler = _handler
    size = handler.queue.qsize() if handler is not None and not handler._stopped else 0
    return {"log_dropped_total": _dropped_total, "log_queue_size": size}


def configure(service: str, *, stream: Any = None, heartbeat: bool = True) -> None:
    """配置根日志与 uvicorn 日志为 JSON 行；幂等。应在 uvicorn 导入应用时调用。"""
    global _handler, _service
    if _handler is not None:
        return
    _service = service
    level = os.environ.get("LOG_LEVEL", "INFO").strip().upper() or "INFO"
    level = {"WARN": "WARNING", "FATAL": "CRITICAL"}.get(level, level)
    if level not in ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"):
        level = "INFO"
    target = logging.StreamHandler(stream or sys.stdout)
    target.setFormatter(JsonFormatter(service))
    handler = _BoundedQueueHandler(target)
    handler.addFilter(_ContextFilter())
    root = logging.getLogger()
    for existing in list(root.handlers):
        root.removeHandler(existing)
    root.addHandler(handler)
    root.setLevel(level)
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        logger = logging.getLogger(name)
        for existing in list(logger.handlers):
            logger.removeHandler(existing)
        logger.propagate = True
        logger.setLevel(level)
    # 访问日志由 InferenceLogMiddleware 承担；uvicorn 访问日志只保留警告以上。
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    uvicorn_error = logging.getLogger("uvicorn.error")
    if not any(isinstance(item, DuplicateAsgiErrorFilter) for item in uvicorn_error.filters):
        uvicorn_error.addFilter(DuplicateAsgiErrorFilter())
    _handler = handler
    if heartbeat:
        start_heartbeat()


def emit_heartbeat() -> None:
    uptime_s = int(time.monotonic() - _started)
    logging.getLogger("gta_ai.heartbeat").info(
        "log_heartbeat uptime_s=%d", uptime_s,
        extra={"event": "log_heartbeat", "fields": {"uptime_s": uptime_s, **queue_stats()}},
    )


def start_heartbeat(interval_seconds: float = HEARTBEAT_INTERVAL_SECONDS) -> None:
    global _heartbeat_thread
    if _heartbeat_thread is not None:
        return

    def run() -> None:
        stop = threading.Event()
        while True:
            with suppress(Exception):  # 心跳失败不能终止线程
                emit_heartbeat()
            stop.wait(interval_seconds)

    _heartbeat_thread = threading.Thread(target=run, name="log-heartbeat", daemon=True)
    _heartbeat_thread.start()


def annotate(**values: Any) -> None:
    """为当前请求的完成事件补充字段（如 batch_size、queue_wait_ms、reason）。

    asyncio.to_thread 会复制上下文，线程内写入的是同一个字典，因此推理线程中也可调用。"""
    context = _context.get()
    if context is not None:
        context["fields"].update(values)


def _header(headers: list[tuple[bytes, bytes]], name: bytes) -> str | None:
    for key, value in headers:
        if key.lower() == name:
            text = value.decode("latin-1").strip()
            return text if _ID_PATTERN.fullmatch(text) else None
    return None


class InferenceLogMiddleware:
    """纯 ASGI 中间件：每个 HTTP 请求一条完成日志，不缓冲请求或响应体。"""

    def __init__(self, app: Any, *, infer_prefix: str = "/v1/",
                 quiet_paths: tuple[str, ...] = ("/health",)) -> None:
        self.app = app
        self._infer_prefix = infer_prefix
        self._quiet_paths = quiet_paths
        self._logger = logging.getLogger("gta_ai.request")

    async def __call__(self, scope: dict, receive: Any, send: Any) -> None:
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return
        headers = list(scope.get("headers") or [])
        context: dict[str, Any] = {"fields": {}, "trace_id": _header(headers, b"x-trace-id")}
        context["request_id"] = (
            _header(headers, b"x-gta-request-id") or _header(headers, b"x-request-id"))
        for name in _TASK_HEADERS:
            task_id = _header(headers, name)
            if task_id:
                context["task_id"] = task_id
                context["fields"]["task_kind"] = name.decode()
                break
        priority = _header(headers, b"x-gta-priority")
        if priority:
            context["fields"]["priority"] = priority[:8]
        token = _context.set(context)
        started = time.perf_counter()
        status: int | None = None

        async def send_wrapper(message: dict) -> None:
            nonlocal status
            if message.get("type") == "http.response.start":
                status = int(message["status"])
            await send(message)

        error: BaseException | None = None
        try:
            await self.app(scope, receive, send_wrapper)
        except BaseException as exc:
            error = exc
            raise
        finally:
            try:
                self._log(scope, context, status, error, (time.perf_counter() - started) * 1000)
            finally:
                _context.reset(token)

    def _log(self, scope: dict, context: dict[str, Any], status: int | None,
             error: BaseException | None, duration_ms: float) -> None:
        path = scope.get("path", "")
        cancelled = error is not None and type(error).__name__ == "CancelledError"
        effective = status
        if effective is None and error is not None and not cancelled:
            effective = 500
        if cancelled:
            result, level = "cancelled", logging.INFO
        elif error is not None or effective is None or effective >= 500:
            result, level = "failed", logging.ERROR
        elif effective >= 400:
            result, level = "rejected", logging.WARNING
        else:
            result = "ok"
            level = logging.DEBUG if path.startswith(self._quiet_paths) else logging.INFO
        inference = path.startswith(self._infer_prefix)
        fields = {"path": path, "method": scope.get("method", ""), "status": effective,
                  **context["fields"]}
        if error is not None:
            fields["error_type"] = type(error).__name__
        exc_info = None
        if error is not None and not cancelled:
            exc_info = (type(error), error, error.__traceback__)
        self._logger.log(
            level, "%s %s -> %s %dms", fields["method"], path, effective, int(duration_ms),
            exc_info=exc_info,
            extra={"event": "model_infer" if inference else "http_request", "result": result,
                   "duration_ms": int(duration_ms), "fields": fields,
                   "trace_id": context.get("trace_id"), "request_id": context.get("request_id"),
                   "task_id": context.get("task_id")},
        )
