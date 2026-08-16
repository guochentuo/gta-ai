from __future__ import annotations

import asyncio
import fcntl
import heapq
import json
import os
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import StreamingResponse

WORKLOAD_PATHS = {
    "/v1/chat/completions",
    "/v1/completions",
    "/v1/responses",
}
HOP_BY_HOP_HEADERS = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailers",
    "transfer-encoding",
    "upgrade",
    "content-length",
}


@dataclass(frozen=True)
class RouterConfig:
    host: str = "127.0.0.1"
    port: int = 8000
    backend_url: str = "http://127.0.0.1:18086"
    state_path: Path = Path("/opt/gta-ai/training/runtime/router-state.json")
    gpu_lock_path: Path = Path("/opt/gta-ai/training/runtime/gpu.lock")
    wake_timeout_seconds: float = 300.0
    backend_poll_seconds: float = 0.25
    request_timeout_seconds: float = 1800.0
    max_concurrent_workloads: int = 1

    @classmethod
    def from_environment(cls) -> RouterConfig:
        return cls(
            host=os.getenv("GTA_AI_ROUTER_HOST", cls.host),
            port=int(os.getenv("GTA_AI_ROUTER_PORT", str(cls.port))),
            backend_url=os.getenv("GTA_AI_ROUTER_BACKEND_URL", cls.backend_url).rstrip("/"),
            state_path=Path(os.getenv("GTA_AI_ROUTER_STATE_PATH", str(cls.state_path))),
            gpu_lock_path=Path(os.getenv("GTA_AI_ROUTER_GPU_LOCK_PATH", str(cls.gpu_lock_path))),
            wake_timeout_seconds=float(
                os.getenv("GTA_AI_ROUTER_WAKE_TIMEOUT_SECONDS", str(cls.wake_timeout_seconds))
            ),
            backend_poll_seconds=float(
                os.getenv("GTA_AI_ROUTER_BACKEND_POLL_SECONDS", str(cls.backend_poll_seconds))
            ),
            request_timeout_seconds=float(
                os.getenv(
                    "GTA_AI_ROUTER_REQUEST_TIMEOUT_SECONDS",
                    str(cls.request_timeout_seconds),
                )
            ),
            max_concurrent_workloads=max(
                1,
                int(
                    os.getenv(
                        "GTA_AI_ROUTER_MAX_CONCURRENT_WORKLOADS",
                        str(cls.max_concurrent_workloads),
                    )
                ),
            ),
        )


class RuntimeState:
    def __init__(self, path: Path) -> None:
        self._path = path
        self._lock = asyncio.Lock()
        self._active_requests = 0
        self._last_workload_at = time.time()

    async def begin(self) -> None:
        async with self._lock:
            self._active_requests += 1
            self._last_workload_at = time.time()
            self._persist()

    async def end(self) -> None:
        async with self._lock:
            self._active_requests = max(0, self._active_requests - 1)
            self._last_workload_at = time.time()
            self._persist()

    async def snapshot(self) -> dict[str, float | int]:
        async with self._lock:
            return {
                "schema_version": 1,
                "active_requests": self._active_requests,
                "last_workload_at": self._last_workload_at,
                "updated_at": time.time(),
            }

    def initialize(self) -> None:
        self._persist()

    def _persist(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self._path.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "active_requests": self._active_requests,
                    "last_workload_at": self._last_workload_at,
                    "updated_at": time.time(),
                },
                separators=(",", ":"),
            )
            + "\n",
            encoding="utf-8",
        )
        temporary.replace(self._path)


class PriorityWorkloadGate:
    """为 max-num-seqs=1 的模型提供稳定的 P0/P1/P9 排队顺序。"""

    def __init__(self, maximum_active: int) -> None:
        self._maximum_active = max(1, maximum_active)
        self._active = 0
        self._sequence = 0
        self._waiting: list[tuple[int, int, asyncio.Future[None]]] = []
        self._lock = asyncio.Lock()

    @staticmethod
    def parse_priority(value: str | None) -> int:
        normalized = (value or "P1").strip().upper()
        if normalized == "P0":
            return 0
        if normalized == "P9":
            return 9
        return 1

    async def acquire(self, priority: int) -> None:
        loop = asyncio.get_running_loop()
        waiter: asyncio.Future[None] | None = None
        async with self._lock:
            if self._active < self._maximum_active and not self._waiting:
                self._active += 1
                return
            waiter = loop.create_future()
            self._sequence += 1
            heapq.heappush(self._waiting, (priority, self._sequence, waiter))
        try:
            await waiter
        except BaseException:
            waiter.cancel()
            raise

    async def release(self) -> None:
        async with self._lock:
            self._active = max(0, self._active - 1)
            while self._waiting and self._active < self._maximum_active:
                _, _, waiter = heapq.heappop(self._waiting)
                if waiter.cancelled():
                    continue
                self._active += 1
                waiter.set_result(None)

    async def snapshot(self) -> dict[str, int]:
        async with self._lock:
            realtime_waiting = sum(
                1 for priority, _, waiter in self._waiting
                if priority <= 1 and not waiter.cancelled()
            )
            historical_waiting = sum(
                1 for priority, _, waiter in self._waiting
                if priority == 9 and not waiter.cancelled()
            )
            return {
                "active_workloads": self._active,
                "realtime_waiting": realtime_waiting,
                "historical_waiting": historical_waiting,
            }


def _forward_headers(headers: httpx.Headers) -> dict[str, str]:
    return {key: value for key, value in headers.items() if key.lower() not in HOP_BY_HOP_HEADERS}


async def _acquire_inference_gate(path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_CREAT | os.O_RDWR, 0o660)
    try:
        await asyncio.to_thread(fcntl.flock, descriptor, fcntl.LOCK_SH)
    except BaseException:
        os.close(descriptor)
        raise
    return descriptor


def _release_inference_gate(descriptor: int | None) -> None:
    if descriptor is None:
        return
    fcntl.flock(descriptor, fcntl.LOCK_UN)
    os.close(descriptor)


def create_router_app(
    config: RouterConfig | None = None,
    *,
    transport: httpx.AsyncBaseTransport | None = None,
) -> FastAPI:
    runtime_config = config or RouterConfig.from_environment()
    runtime_state = RuntimeState(runtime_config.state_path)
    workload_gate = PriorityWorkloadGate(runtime_config.max_concurrent_workloads)

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        runtime_state.initialize()
        yield

    application = FastAPI(
        title="gta-ai inference router",
        description="Inference-first router for preemptible idle fine-tuning.",
        lifespan=lifespan,
    )

    async def backend_ready(client: httpx.AsyncClient) -> bool:
        try:
            response = await client.get(f"{runtime_config.backend_url}/v1/models", timeout=1)
            return response.status_code == 200
        except httpx.HTTPError:
            return False

    async def wait_for_backend(client: httpx.AsyncClient) -> None:
        deadline = time.monotonic() + runtime_config.wake_timeout_seconds
        while time.monotonic() < deadline:
            if await backend_ready(client):
                return
            await asyncio.sleep(runtime_config.backend_poll_seconds)
        raise HTTPException(
            status_code=503,
            detail="推理服务未能在训练抢占窗口内恢复",
            headers={"Retry-After": "5"},
        )

    @application.get("/health/live")
    async def live() -> dict[str, str]:
        return {"state": "ok"}

    @application.get("/_gta/runtime")
    async def runtime() -> dict[str, float | int]:
        state = await runtime_state.snapshot()
        state.update(await workload_gate.snapshot())
        return state

    @application.api_route(
        "/{path:path}",
        methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS", "HEAD"],
    )
    async def proxy(path: str, request: Request) -> Response:
        upstream_path = "/" + path
        is_workload = request.method in {"POST", "PUT", "PATCH"} and (
            upstream_path in WORKLOAD_PATHS or upstream_path.startswith("/v1/")
        )
        if is_workload:
            await runtime_state.begin()

        inference_gate: int | None = None
        workload_gate_acquired = False
        timeout = httpx.Timeout(runtime_config.request_timeout_seconds, connect=2)
        client = httpx.AsyncClient(timeout=timeout, transport=transport)
        try:
            if is_workload:
                await workload_gate.acquire(
                    workload_gate.parse_priority(request.headers.get("x-gta-priority"))
                )
                workload_gate_acquired = True
                # 共享锁覆盖整个推理响应。训练持有独占锁时, 新请求先登记为 active,
                # 然后等待调度器抢占训练并释放 GPU; 绝不会撞上正在退出的 vLLM。
                inference_gate = await _acquire_inference_gate(runtime_config.gpu_lock_path)
                await wait_for_backend(client)
            upstream_request = client.build_request(
                request.method,
                f"{runtime_config.backend_url}{upstream_path}",
                params=request.query_params,
                headers={
                    key: value
                    for key, value in request.headers.items()
                    if key.lower() not in HOP_BY_HOP_HEADERS and key.lower() != "host"
                },
                # 请求体逐字节转发; 不添加身份消息; 也不修改 model 字段。
                content=await request.body(),
            )
            upstream = await client.send(upstream_request, stream=True)
        except HTTPException:
            await client.aclose()
            _release_inference_gate(inference_gate)
            if workload_gate_acquired:
                await workload_gate.release()
            if is_workload:
                await runtime_state.end()
            raise
        except httpx.HTTPError as exc:
            await client.aclose()
            _release_inference_gate(inference_gate)
            if workload_gate_acquired:
                await workload_gate.release()
            if is_workload:
                await runtime_state.end()
            raise HTTPException(
                status_code=502, detail=f"推理后端连接失败: {type(exc).__name__}"
            ) from exc

        async def stream() -> AsyncIterator[bytes]:
            try:
                async for chunk in upstream.aiter_raw():
                    yield chunk
            finally:
                await upstream.aclose()
                await client.aclose()
                _release_inference_gate(inference_gate)
                if workload_gate_acquired:
                    await workload_gate.release()
                if is_workload:
                    await runtime_state.end()

        return StreamingResponse(
            stream(),
            status_code=upstream.status_code,
            headers=_forward_headers(upstream.headers),
            media_type=upstream.headers.get("content-type"),
        )

    return application


app = create_router_app()


def main() -> None:
    config = RouterConfig.from_environment()
    uvicorn.run(app, host=config.host, port=config.port, log_level="info")
