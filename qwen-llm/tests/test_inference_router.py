from __future__ import annotations

import asyncio
import fcntl
import json
from pathlib import Path

import httpx
import pytest
from router.app import (
    SYSTEM_PROMPT,
    LeaseAcquireRequest,
    PersistentGpuAdmission,
    PriorityWorkloadGate,
    RouterConfig,
    create_router_app,
)


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
async def test_cancelled_priority_waiter_is_removed_immediately() -> None:
    gate = PriorityWorkloadGate(1)
    await gate.acquire(9)
    waiting = asyncio.create_task(gate.acquire(9))
    await asyncio.sleep(0)
    assert (await gate.snapshot())["historical_waiting"] == 1
    waiting.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiting
    assert (await gate.snapshot())["historical_waiting"] == 0
    await gate.release()


@pytest.mark.asyncio
async def test_router_injects_owned_persona(tmp_path: Path) -> None:
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
        admission_db_path=tmp_path / "admission.sqlite3",
        wake_timeout_seconds=1,
        backend_poll_seconds=0.01,
    )
    app = create_router_app(config, transport=transport)
    payload = {
        "model": "Qwen/Qwen3.8-27B-FP8",
        "messages": [{"role": "user", "content": "你是谁?"}],
    }
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://router"
    ) as client:
        response = await client.post("/v1/chat/completions", json=payload)
        await response.aread()

    assert response.status_code == 200
    assert len(captured) == 1
    forwarded = json.loads(captured[0])
    assert forwarded["model"] == payload["model"]
    assert forwarded["messages"] == [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "你是谁?"},
    ]


@pytest.mark.asyncio
async def test_router_replaces_caller_system_identity(tmp_path: Path) -> None:
    captured: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "GET" and request.url.path == "/v1/models":
            return httpx.Response(200, json={"data": [{"id": "base"}]})
        captured.append(request)
        return httpx.Response(200, stream=_MockStream(b"done"))

    app = create_router_app(
        RouterConfig(
            backend_url="http://backend",
            state_path=tmp_path / "state.json",
            gpu_lock_path=tmp_path / "gpu.lock",
            admission_db_path=tmp_path / "admission.sqlite3",
        ),
        transport=httpx.MockTransport(handler),
    )
    payload = {
        "model": "Qwen/Qwen3.8-27B-FP8",
        "messages": [
            {"role": "system", "content": "你是通义千问"},
            {"role": "user", "content": "你是谁?"},
        ],
        "stream": True,
    }
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://router"
    ) as client:
        response = await client.post(
            "/v1/chat/completions",
            json=payload,
            headers={"X-GTA-Persona": "green-travel"},
        )
        assert await response.aread() == b"done"

    assert response.status_code == 200
    assert len(captured) == 1
    forwarded = json.loads(captured[0].content)
    assert forwarded["messages"] == [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "你是谁?"},
    ]
    assert forwarded["stream"] is True
    assert captured[0].headers["x-gta-persona"] == "green-travel"


@pytest.mark.asyncio
async def test_router_rejects_invalid_chat_json(
    tmp_path: Path,
) -> None:
    captured: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(200, stream=_MockStream(b"done"))

    app = create_router_app(
        RouterConfig(
            backend_url="http://backend",
            state_path=tmp_path / "state.json",
            gpu_lock_path=tmp_path / "gpu.lock",
            admission_db_path=tmp_path / "admission.sqlite3",
        ),
        transport=httpx.MockTransport(handler),
    )
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://router"
    ) as client:
        response = await client.post(
            "/v1/chat/completions",
            content=b"not-json",
            headers={"X-GTA-Persona": "green-travel"},
        )

    assert response.status_code == 422
    assert not captured


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
            admission_db_path=tmp_path / "admission.sqlite3",
        ),
        transport=httpx.MockTransport(handler),
    )
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://router"
    ) as client:
        response = await client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "test"}]},
        )
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
            admission_db_path=tmp_path / "admission.sqlite3",
        ),
        transport=httpx.MockTransport(handler),
    )
    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://router"
        ) as client:
            request = asyncio.create_task(
                client.post(
                    "/v1/chat/completions",
                    json={"messages": [{"role": "user", "content": "test"}]},
                )
            )
            await asyncio.sleep(0.05)
            assert not request.done()
            state = json.loads(state_path.read_text(encoding="utf-8"))
            assert state["active_requests"] == 1
            fcntl.flock(training_lock, fcntl.LOCK_UN)
            response = await asyncio.wait_for(request, timeout=1)
            assert response.content == b"done"
    finally:
        training_lock.close()


@pytest.mark.asyncio
async def test_explicit_request_cancel_releases_all_router_state(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "GET":
            return httpx.Response(200, json={"data": []})
        return httpx.Response(200, stream=_MockStream(b"done"))

    lock_path = tmp_path / "gpu.lock"
    lock_path.touch()
    training_lock = lock_path.open("a+")
    fcntl.flock(training_lock, fcntl.LOCK_EX)
    app = create_router_app(
        RouterConfig(
            backend_url="http://backend",
            state_path=tmp_path / "state.json",
            gpu_lock_path=lock_path,
            admission_db_path=tmp_path / "admission.sqlite3",
            request_timeout_seconds=5,
        ),
        transport=httpx.MockTransport(handler),
        memory_provider=lambda: (1000, 8000),
    )
    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://router"
        ) as client:
            pending = asyncio.create_task(
                client.post(
                    "/v1/chat/completions",
                        json={"messages": [{"role": "user", "content": "test"}]},
                    headers={"X-GTA-Request-ID": "task:cancel-me"},
                )
            )
            await asyncio.sleep(0)
            for _ in range(50):
                state = (await client.get("/_gta/runtime")).json()
                if state["tracked_requests"] == 1:
                    break
                await asyncio.sleep(0.01)
            duplicate = await client.post(
                "/v1/chat/completions",
                json={"messages": [{"role": "user", "content": "test"}]},
                headers={"X-GTA-Request-ID": "task:cancel-me"},
            )
            assert duplicate.status_code == 409
            cancelled = await client.delete("/_gta/requests/task:cancel-me")
            assert cancelled.json() == {
                "request_id": "task:cancel-me",
                "cancelled": True,
            }
            response = await asyncio.wait_for(pending, timeout=1)
            assert response.status_code == 499
            assert (await client.delete("/_gta/requests/task:cancel-me")).json()[
                "cancelled"
            ] is False
            state = (await client.get("/_gta/runtime")).json()
            admission = (await client.get("/_gta/admission")).json()
            assert state["active_requests"] == 0
            assert state["active_workloads"] == 0
            assert state["tracked_requests"] == 0
            assert state["historical_waiting"] == 0
            assert admission["leases"] == []
            assert admission["waiters"] == []
    finally:
        fcntl.flock(training_lock, fcntl.LOCK_UN)
        training_lock.close()


@pytest.mark.asyncio
async def test_gpu_lease_survives_manager_restart_and_expires(tmp_path: Path) -> None:
    database = tmp_path / "admission.sqlite3"
    manager = PersistentGpuAdmission(
        database,
        poll_seconds=0.01,
        minimum_free_mb=100,
        memory_provider=lambda: (1000, 8000),
    )
    lease = await manager.acquire(
        LeaseAcquireRequest(
            owner="worker:feature:1",
            workload_class="REALTIME_FEATURE",
            requested_memory_mb=2000,
            ttl_seconds=5,
        )
    )
    restarted = PersistentGpuAdmission(
        database,
        poll_seconds=0.01,
        minimum_free_mb=100,
        memory_provider=lambda: (1000, 8000),
    )
    snapshot = await restarted.snapshot()
    assert [item["lease_id"] for item in snapshot["leases"]] == [lease["lease_id"]]
    assert await restarted.release(str(lease["lease_id"]), "worker:feature:1")
    assert (await restarted.snapshot())["leases"] == []


@pytest.mark.asyncio
async def test_realtime_waiter_runs_before_historical_waiter(tmp_path: Path) -> None:
    manager = PersistentGpuAdmission(
        tmp_path / "admission.sqlite3",
        poll_seconds=0.01,
        minimum_free_mb=100,
        memory_provider=lambda: (42000, 4000),
    )
    active = await manager.acquire(
        LeaseAcquireRequest(
            owner="active:understanding",
            workload_class="REALTIME_UNDERSTANDING",
            ttl_seconds=30,
        )
    )
    order: list[str] = []

    async def acquire_named(name: str, workload_class: str) -> dict[str, object]:
        lease = await manager.acquire(
            LeaseAcquireRequest(
                owner=name,
                workload_class=workload_class,
                ttl_seconds=30,
                wait_seconds=2,
            )
        )
        order.append(name)
        return lease

    historical = asyncio.create_task(
        acquire_named("history", "HISTORICAL_UNDERSTANDING")
    )
    await asyncio.sleep(0.02)
    realtime = asyncio.create_task(
        acquire_named("realtime", "REALTIME_UNDERSTANDING")
    )
    await asyncio.sleep(0.02)
    await manager.release(str(active["lease_id"]), "active:understanding")
    realtime_lease = await asyncio.wait_for(realtime, timeout=1)
    assert order == ["realtime"]
    await manager.release(str(realtime_lease["lease_id"]), "realtime")
    historical_lease = await asyncio.wait_for(historical, timeout=1)
    assert order == ["realtime", "history"]
    await manager.release(str(historical_lease["lease_id"]), "history")


@pytest.mark.asyncio
async def test_admission_http_contract_rejects_unknown_class(tmp_path: Path) -> None:
    app = create_router_app(
        RouterConfig(
            backend_url="http://backend",
            state_path=tmp_path / "state.json",
            gpu_lock_path=tmp_path / "gpu.lock",
            admission_db_path=tmp_path / "admission.sqlite3",
        ),
        transport=httpx.MockTransport(lambda _: httpx.Response(200, json={})),
        memory_provider=lambda: (1000, 8000),
    )
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://router"
    ) as client:
        bad = await client.post(
            "/_gta/admission/leases",
            json={"owner": "worker:1", "workload_class": "UNKNOWN"},
        )
        assert bad.status_code == 422
        good = await client.post(
            "/_gta/admission/leases",
            json={
                "owner": "worker:1",
                "workload_class": "REALTIME_FEATURE",
                "requested_memory_mb": 1000,
            },
        )
        assert good.status_code == 200
        lease = good.json()
        released = await client.delete(
            f"/_gta/admission/leases/{lease['lease_id']}",
            params={"owner": "worker:1"},
        )
        assert released.json() == {"released": True}


@pytest.mark.asyncio
async def test_single_gpu_admission_never_runs_nvenc_whisper_or_27b_together(
    tmp_path: Path,
) -> None:
    manager = PersistentGpuAdmission(
        tmp_path / "admission.sqlite3",
        poll_seconds=0.01,
        minimum_free_mb=2048,
        maximum_active=1,
        memory_provider=lambda: (36591, 8870),
    )
    playback = await manager.acquire(
        LeaseAcquireRequest(
            owner="nvenc:1",
            workload_class="REALTIME_PLAYBACK",
            requested_memory_mb=1024,
            ttl_seconds=30,
        )
    )
    with pytest.raises(TimeoutError):
        await manager.acquire(
            LeaseAcquireRequest(
                owner="whisper:1",
                workload_class="REALTIME_FEATURE",
                requested_memory_mb=6144,
                ttl_seconds=30,
            )
        )
    await manager.release(str(playback["lease_id"]), "nvenc:1")
