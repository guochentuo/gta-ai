"""service_logging：JSON格式、完成事件、级别、关联ID、脱敏、心跳。只依赖 fastapi，不加载模型。"""

from __future__ import annotations

import importlib.util
import io
import json
import logging
import os
from pathlib import Path

import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

PROJECT_DIR = Path(__file__).resolve().parents[1]


@pytest.fixture()
def service_logging():
    spec = importlib.util.spec_from_file_location(
        f"service_logging_{PROJECT_DIR.name.replace('-', '_')}", PROJECT_DIR / "service_logging.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    stream = io.StringIO()
    root = logging.getLogger()
    saved = (list(root.handlers), root.level)
    module.configure("gta-ai-test", stream=stream, heartbeat=False)
    module.test_stream = stream
    try:
        yield module
    finally:
        module._handler.close()
        for handler in list(root.handlers):
            root.removeHandler(handler)
        for handler in saved[0]:
            root.addHandler(handler)
        root.setLevel(saved[1])


def _lines(module) -> list[dict]:
    module._handler.flush()
    return [json.loads(line) for line in module.test_stream.getvalue().splitlines() if line.strip()]


def _app(module):
    from fastapi import HTTPException, Request
    from fastapi.exception_handlers import http_exception_handler

    app = fastapi.FastAPI()
    app.add_middleware(module.InferenceLogMiddleware)

    @app.exception_handler(HTTPException)
    async def handle(request: Request, exc: HTTPException):
        module.annotate(reason=str(exc.detail))
        return await http_exception_handler(request, exc)

    @app.post("/v1/ok")
    async def ok():
        module.annotate(batch_size=3)
        return {"ok": True}

    @app.post("/v1/bad")
    async def bad():
        module.annotate(batch_size=99)
        raise HTTPException(status_code=400, detail="批次大小不符合限制")

    @app.post("/v1/crash")
    async def crash():
        raise RuntimeError("model failed api_key=secret123 https://h.example/x?token=abc")

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    return app


def test_success_emits_model_infer_with_ids(service_logging, monkeypatch):
    client = TestClient(_app(service_logging))
    response = client.post("/v1/ok", headers={
        "X-Trace-Id": "tr-1", "X-GTA-Task-ID": "video-9:frame-2", "X-GTA-Priority": "P9"})
    assert response.status_code == 200
    line = next(item for item in _lines(service_logging) if item.get("event") == "model_infer")
    assert line["level"] == "INFO" and line["result"] == "ok"
    assert line["trace_id"] == "tr-1" and line["task_id"] == "video-9:frame-2"
    assert line["fields"]["batch_size"] == 3 and line["fields"]["status"] == 200
    assert line["fields"]["priority"] == "P9" and line["fields"]["task_kind"] == "x-gta-task-id"
    assert isinstance(line["duration_ms"], int)
    for key in ("@timestamp", "env", "service", "project", "node", "host", "deploy_sha", "logger"):
        assert key in line
    assert line["service"] == "gta-ai-test" and line["project"] == "gta-ai"


def test_env_node_sha_default_to_unknown(service_logging):
    client = TestClient(_app(service_logging))
    client.post("/v1/ok")
    line = _lines(service_logging)[-1]
    # 测试环境未注入 GTA_ENV/GTA_NODE/GTA_DEPLOY_SHA 时不得猜测
    assert line["env"] == (os.environ.get("GTA_ENV", "").strip() or "unknown")


def test_client_error_is_warn_with_reason(service_logging):
    client = TestClient(_app(service_logging))
    assert client.post("/v1/bad", headers={"X-Request-Id": "req-7"}).status_code == 400
    line = next(item for item in _lines(service_logging) if item.get("event") == "model_infer")
    assert line["level"] == "WARN" and line["result"] == "rejected"
    assert line["fields"]["reason"] == "批次大小不符合限制"
    assert line["task_id"] == "req-7" and line["request_id"] == "req-7"
    assert "error" not in line


def test_exception_is_one_error_with_stack_and_redacted(service_logging):
    client = TestClient(_app(service_logging), raise_server_exceptions=False)
    assert client.post("/v1/crash").status_code == 500
    lines = _lines(service_logging)
    errors = [item for item in lines if item["level"] == "ERROR"]
    assert len(errors) == 1
    line = errors[0]
    assert line["event"] == "model_infer" and line["result"] == "failed"
    assert line["fields"]["status"] == 500 and line["fields"]["error_type"] == "RuntimeError"
    assert "Traceback" in line["error"]
    assert "secret123" not in line["error"] and "token=abc" not in line["error"]
    assert "token=***" in line["error"]


def test_health_success_is_quiet(service_logging):
    client = TestClient(_app(service_logging))
    client.get("/health")
    lines = _lines(service_logging)
    assert not [item for item in lines if item.get("fields", {}).get("path") == "/health"]


def test_uvicorn_duplicate_asgi_error_demoted(service_logging):
    logging.getLogger("uvicorn.error").error("Exception in ASGI application\n")
    logging.getLogger("uvicorn.error").error("Application startup failed")
    messages = [item["message"] for item in _lines(service_logging)]
    assert "Application startup failed" in messages
    assert not any(message.startswith("Exception in ASGI application") for message in messages)


def test_heartbeat_line(service_logging):
    service_logging.emit_heartbeat()
    line = next(item for item in _lines(service_logging) if item.get("event") == "log_heartbeat")
    assert line["level"] == "INFO"
    assert {"uptime_s", "log_dropped_total", "log_queue_size"} <= set(line["fields"])


def test_queue_full_drops_and_counts(service_logging):
    handler = service_logging._BoundedQueueHandler(logging.NullHandler(), capacity=1)
    handler.listener.stop()
    before = service_logging._dropped_total
    for index in range(4):
        handler.handle(logging.LogRecord("t", logging.INFO, __file__, 1, "m%d", (index,), None))
    assert service_logging._dropped_total - before >= 3
    handler._stopped = True


def test_copies_are_identical_across_projects():
    copies = {path.read_bytes() for path in PROJECT_DIR.parent.glob("*/service_logging.py")}
    assert len(copies) == 1
