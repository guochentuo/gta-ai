from __future__ import annotations

import asyncio
import fcntl
import json
from pathlib import Path

import httpx
import pytest

from gta_ai.inference_router import PriorityWorkloadGate, RouterConfig, create_router_app


class _MockStream(httpx.AsyncByteStream):
    def __init__(self, content: bytes) -> None:
        self.content = content

    async def __aiter__(self):
        yield self.content


@pytest.mark.asyncio
async def test_priority_gate_runs_realtime_before_waiting_history() -> None:
    gate = PriorityWorkloadGate(1)
    await gate.acquire(9)
    order: list[str] = []

    async def wait_for_slot(name: str, priority: int) -> None:
        await gate.acquire(priority)
        order.append(name)

    historical = asyncio.create_task(wait_for_slot("history", 9))
    realtime = asyncio.create_task(wait_for_slot("realtime", 1))
    await asyncio.sleep(0)
    await gate.release()
    await asyncio.wait_for(realtime, timeout=1)
    assert order == ["realtime"]
    await gate.release()
    await asyncio.wait_for(historical, timeout=1)
    assert order == ["realtime", "history"]
    await gate.release()


@pytest.mark.asyncio
async def test_router_forwards_request_bytes_without_identity_injection(tmp_path: Path) -> None:
    captured: list[bytes] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "GET" and request.url.path == "/v1/models":
            return httpx.Response(200, json={"data": [{"id": "base"}]})
        captured.append(request.content)
        return httpx.Response(
            200,
            headers={"content-type": "application/json"},
            stream=_MockStream(b'{"choices":[{"message":{"content":"ok"}}]}'),
        )

    transport = httpx.MockTransport(handler)
    config = RouterConfig(
        backend_url="http://backend",
        state_path=tmp_path / "state.json",
        gpu_lock_path=tmp_path / "gpu.lock",
        wake_timeout_seconds=1,
        backend_poll_seconds=0.01,
    )
    app = create_router_app(config, transport=transport)
    payload = {
        "model": "Qwen/Qwen3.6-27B-FP8",
        "messages": [{"role": "user", "content": "你是谁?"}],
    }
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://router"
    ) as client:
        response = await client.post("/v1/chat/completions", json=payload)
        await response.aread()

    assert response.status_code == 200
    assert len(captured) == 1
    assert json.loads(captured[0]) == payload
    assert "system" not in captured[0].decode("utf-8")


@pytest.mark.asyncio
async def test_router_records_workload_lifecycle(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "GET":
            return httpx.Response(200, json={"data": []})
        return httpx.Response(200, stream=_MockStream(b"done"))

    state_path = tmp_path / "state.json"
    app = create_router_app(
        RouterConfig(
            backend_url="http://backend",
            state_path=state_path,
            gpu_lock_path=tmp_path / "gpu.lock",
        ),
        transport=httpx.MockTransport(handler),
    )
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://router"
    ) as client:
        response = await client.post("/v1/chat/completions", content=b"{}")
        assert await response.aread() == b"done"

    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert state["active_requests"] == 0
    assert state["last_workload_at"] > 0


@pytest.mark.asyncio
async def test_request_registers_active_before_waiting_for_training_lock(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "GET":
            return httpx.Response(200, json={"data": []})
        return httpx.Response(200, stream=_MockStream(b"done"))

    state_path = tmp_path / "state.json"
    lock_path = tmp_path / "gpu.lock"
    lock_path.touch()
    training_lock = lock_path.open("a+")
    fcntl.flock(training_lock, fcntl.LOCK_EX)
    app = create_router_app(
        RouterConfig(
            backend_url="http://backend",
            state_path=state_path,
            gpu_lock_path=lock_path,
        ),
        transport=httpx.MockTransport(handler),
    )
    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://router"
        ) as client:
            request = asyncio.create_task(client.post("/v1/chat/completions", content=b"{}"))
            await asyncio.sleep(0.05)
            assert not request.done()
            state = json.loads(state_path.read_text(encoding="utf-8"))
            assert state["active_requests"] == 1
            fcntl.flock(training_lock, fcntl.LOCK_UN)
            response = await asyncio.wait_for(request, timeout=1)
            assert response.content == b"done"
    finally:
        training_lock.close()
