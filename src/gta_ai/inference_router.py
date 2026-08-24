from __future__ import annotations

import asyncio
import fcntl
import hashlib
import heapq
import json
import os
import re
import sqlite3
import subprocess
import time
import uuid
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

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
REQUEST_ID_PATTERN = re.compile(r"^[A-Za-z0-9._:@/-]{1,160}$")


@dataclass(frozen=True)
class RouterConfig:
    host: str = "127.0.0.1"
    port: int = 8000
    backend_url: str = "http://127.0.0.1:18086"
    state_path: Path = Path("/opt/gta-ai/router/state/router-state.json")
    gpu_lock_path: Path = Path("/opt/gta-ai/router/state/gpu.lock")
    wake_timeout_seconds: float = 300.0
    backend_poll_seconds: float = 0.25
    request_timeout_seconds: float = 1800.0
    max_concurrent_workloads: int = 1
    admission_db_path: Path = Path("/opt/gta-ai/router/state/gpu-admission.sqlite3")
    admission_poll_seconds: float = 0.1
    admission_minimum_free_mb: int = 2048
    admission_max_active: int = 1

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
            admission_db_path=Path(
                os.getenv("GTA_AI_ADMISSION_DB_PATH", str(cls.admission_db_path))
            ),
            admission_poll_seconds=max(
                0.02,
                float(
                    os.getenv(
                        "GTA_AI_ADMISSION_POLL_SECONDS",
                        str(cls.admission_poll_seconds),
                    )
                ),
            ),
            admission_minimum_free_mb=max(
                0,
                int(
                    os.getenv(
                        "GTA_AI_ADMISSION_MINIMUM_FREE_MB",
                        str(cls.admission_minimum_free_mb),
                    )
                ),
            ),
            admission_max_active=max(
                1,
                int(
                    os.getenv(
                        "GTA_AI_ADMISSION_MAX_ACTIVE",
                        str(cls.admission_max_active),
                    )
                ),
            ),
        )


WORKLOAD_PRIORITIES = {
    "REALTIME_PLAYBACK": 0,
    "REALTIME_FEATURE": 1,
    "REALTIME_UNDERSTANDING": 2,
    "HISTORICAL_FEATURE": 9,
    "HISTORICAL_UNDERSTANDING": 10,
}


class LeaseAcquireRequest(BaseModel):
    owner: str = Field(min_length=1, max_length=160, pattern=r"^[A-Za-z0-9._:@/-]+$")
    workload_class: str
    requested_memory_mb: int = Field(default=0, ge=0, le=46068)
    ttl_seconds: int = Field(default=1800, ge=5, le=7200)
    wait_seconds: float = Field(default=0.0, ge=0.0, le=7200.0)


class LeaseHeartbeatRequest(BaseModel):
    owner: str = Field(min_length=1, max_length=160, pattern=r"^[A-Za-z0-9._:@/-]+$")
    ttl_seconds: int = Field(default=1800, ge=5, le=7200)


def _gpu_memory_snapshot() -> tuple[int, int]:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.used,memory.free",
                "--format=csv,noheader,nounits",
                "--id=0",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=2,
        )
        first = result.stdout.strip().splitlines()[0]
        used, free = (int(part.strip()) for part in first.split(",", maxsplit=1))
        return used, free
    except (OSError, ValueError, IndexError, subprocess.SubprocessError):
        return 0, 0


class PersistentGpuAdmission:
    """Single-GPU cooperative admission with persistent expiring leases.

    It never kills active work.  A realtime waiter prevents new historical
    leases, while an already running historical lease is allowed to finish.
    """

    def __init__(
        self,
        path: Path,
        *,
        poll_seconds: float,
        minimum_free_mb: int,
        maximum_active: int = 1,
        memory_provider: Callable[[], tuple[int, int]] = _gpu_memory_snapshot,
    ) -> None:
        self._path = path
        self._poll_seconds = poll_seconds
        self._minimum_free_mb = minimum_free_mb
        self._maximum_active = max(1, maximum_active)
        self._memory_provider = memory_provider
        self._lock = asyncio.Lock()
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self._path, timeout=5)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.executescript(
                """
                PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS gpu_lease (
                    lease_id TEXT PRIMARY KEY,
                    owner TEXT NOT NULL,
                    workload_class TEXT NOT NULL,
                    priority INTEGER NOT NULL,
                    requested_memory_mb INTEGER NOT NULL,
                    acquired_at REAL NOT NULL,
                    heartbeat_at REAL NOT NULL,
                    expires_at REAL NOT NULL
                );
                CREATE UNIQUE INDEX IF NOT EXISTS ux_gpu_lease_owner
                    ON gpu_lease(owner);
                CREATE TABLE IF NOT EXISTS gpu_waiter (
                    waiter_id TEXT PRIMARY KEY,
                    owner TEXT NOT NULL,
                    workload_class TEXT NOT NULL,
                    priority INTEGER NOT NULL,
                    requested_memory_mb INTEGER NOT NULL,
                    created_at REAL NOT NULL,
                    expires_at REAL NOT NULL
                );
                CREATE INDEX IF NOT EXISTS ix_gpu_waiter_order
                    ON gpu_waiter(priority, created_at, waiter_id);
                """
            )

    @staticmethod
    def _priority(workload_class: str) -> tuple[str, int]:
        normalized = workload_class.strip().upper()
        if normalized not in WORKLOAD_PRIORITIES:
            raise ValueError(f"unsupported workload_class: {workload_class}")
        return normalized, WORKLOAD_PRIORITIES[normalized]

    @staticmethod
    def _is_historical(priority: int) -> bool:
        return priority >= WORKLOAD_PRIORITIES["HISTORICAL_FEATURE"]

    @staticmethod
    def _class_limit(workload_class: str) -> int:
        if workload_class in {"REALTIME_UNDERSTANDING", "HISTORICAL_UNDERSTANDING"}:
            return 1
        if workload_class in {"REALTIME_PLAYBACK", "HISTORICAL_FEATURE"}:
            return 2
        return 1

    @staticmethod
    def _purge(connection: sqlite3.Connection, now: float) -> None:
        connection.execute("DELETE FROM gpu_lease WHERE expires_at <= ?", (now,))
        connection.execute("DELETE FROM gpu_waiter WHERE expires_at <= ?", (now,))

    def _can_admit(
        self,
        connection: sqlite3.Connection,
        *,
        waiter_id: str,
        workload_class: str,
        priority: int,
        requested_memory_mb: int,
    ) -> bool:
        first = connection.execute(
            "SELECT waiter_id, priority FROM gpu_waiter "
            "ORDER BY priority, created_at, waiter_id LIMIT 1"
        ).fetchone()
        if first is None or first["waiter_id"] != waiter_id:
            return False
        if self._is_historical(priority):
            realtime = connection.execute(
                "SELECT 1 FROM gpu_waiter WHERE priority < ? AND waiter_id <> ? LIMIT 1",
                (WORKLOAD_PRIORITIES["HISTORICAL_FEATURE"], waiter_id),
            ).fetchone()
            if realtime is not None:
                return False
        active_same = connection.execute(
            "SELECT COUNT(*) FROM gpu_lease WHERE workload_class = ?",
            (workload_class,),
        ).fetchone()[0]
        if active_same >= self._class_limit(workload_class):
            return False
        active_total = connection.execute("SELECT COUNT(*) FROM gpu_lease").fetchone()[0]
        if active_total >= self._maximum_active:
            return False
        if workload_class.endswith("UNDERSTANDING"):
            active_understanding = connection.execute(
                "SELECT COUNT(*) FROM gpu_lease WHERE workload_class LIKE '%UNDERSTANDING'"
            ).fetchone()[0]
            if active_understanding >= 1:
                return False
        active_reserved_mb = connection.execute(
            "SELECT COALESCE(SUM(requested_memory_mb),0) FROM gpu_lease"
        ).fetchone()[0]
        _, free_mb = self._memory_provider()
        return free_mb == 0 or free_mb >= (
            requested_memory_mb + active_reserved_mb + self._minimum_free_mb
        )

    async def acquire(self, request: LeaseAcquireRequest) -> dict[str, object]:
        workload_class, priority = self._priority(request.workload_class)
        waiter_id = uuid.uuid4().hex
        now = time.time()
        deadline = time.monotonic() + request.wait_seconds
        async with self._lock:
            with self._connect() as connection:
                self._purge(connection, now)
                existing = connection.execute(
                    "SELECT * FROM gpu_lease WHERE owner = ?", (request.owner,)
                ).fetchone()
                if existing is not None:
                    return dict(existing)
                connection.execute(
                    "INSERT INTO gpu_waiter VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (
                        waiter_id,
                        request.owner,
                        workload_class,
                        priority,
                        request.requested_memory_mb,
                        now,
                        now + max(request.wait_seconds + 5.0, 10.0),
                    ),
                )
        try:
            while True:
                async with self._lock:
                    now = time.time()
                    with self._connect() as connection:
                        self._purge(connection, now)
                        if self._can_admit(
                            connection,
                            waiter_id=waiter_id,
                            workload_class=workload_class,
                            priority=priority,
                            requested_memory_mb=request.requested_memory_mb,
                        ):
                            lease_id = uuid.uuid4().hex
                            expires_at = now + request.ttl_seconds
                            connection.execute(
                                "INSERT INTO gpu_lease VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                                (
                                    lease_id,
                                    request.owner,
                                    workload_class,
                                    priority,
                                    request.requested_memory_mb,
                                    now,
                                    now,
                                    expires_at,
                                ),
                            )
                            connection.execute(
                                "DELETE FROM gpu_waiter WHERE waiter_id = ?", (waiter_id,)
                            )
                            return {
                                "lease_id": lease_id,
                                "owner": request.owner,
                                "workload_class": workload_class,
                                "priority": priority,
                                "requested_memory_mb": request.requested_memory_mb,
                                "acquired_at": now,
                                "heartbeat_at": now,
                                "expires_at": expires_at,
                            }
                if request.wait_seconds <= 0 or time.monotonic() >= deadline:
                    raise TimeoutError("GPU admission capacity is not currently available")
                await asyncio.sleep(self._poll_seconds)
        finally:
            async with self._lock:
                with self._connect() as connection:
                    connection.execute("DELETE FROM gpu_waiter WHERE waiter_id = ?", (waiter_id,))

    async def heartbeat(
        self, lease_id: str, request: LeaseHeartbeatRequest
    ) -> dict[str, object]:
        async with self._lock:
            now = time.time()
            with self._connect() as connection:
                self._purge(connection, now)
                row = connection.execute(
                    "SELECT * FROM gpu_lease WHERE lease_id = ? AND owner = ?",
                    (lease_id, request.owner),
                ).fetchone()
                if row is None:
                    raise KeyError(lease_id)
                expires_at = now + request.ttl_seconds
                connection.execute(
                    "UPDATE gpu_lease SET heartbeat_at = ?, expires_at = ? WHERE lease_id = ?",
                    (now, expires_at, lease_id),
                )
                result = dict(row)
                result.update({"heartbeat_at": now, "expires_at": expires_at})
                return result

    async def release(self, lease_id: str, owner: str) -> bool:
        async with self._lock:
            with self._connect() as connection:
                cursor = connection.execute(
                    "DELETE FROM gpu_lease WHERE lease_id = ? AND owner = ?",
                    (lease_id, owner),
                )
                return cursor.rowcount == 1

    async def snapshot(self) -> dict[str, object]:
        async with self._lock:
            now = time.time()
            with self._connect() as connection:
                self._purge(connection, now)
                leases = [dict(row) for row in connection.execute(
                    "SELECT * FROM gpu_lease ORDER BY priority, acquired_at"
                )]
                waiters = [dict(row) for row in connection.execute(
                    "SELECT * FROM gpu_waiter ORDER BY priority, created_at"
                )]
            used_mb, free_mb = self._memory_provider()
            return {
                "schema_version": 1,
                "gpu_memory_used_mb": used_mb,
                "gpu_memory_free_mb": free_mb,
                "minimum_free_mb": self._minimum_free_mb,
                "leases": leases,
                "waiters": waiters,
            }


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
        priorities = {"P0": 0, "P1": 1, "P2": 2, "P9": 9, "P10": 10}
        return priorities.get(normalized, 1)

    @staticmethod
    def workload_class(priority: int) -> str:
        classes = {
            0: "REALTIME_PLAYBACK",
            1: "REALTIME_FEATURE",
            2: "REALTIME_UNDERSTANDING",
            9: "HISTORICAL_FEATURE",
            10: "HISTORICAL_UNDERSTANDING",
        }
        return classes.get(priority, "REALTIME_FEATURE")

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
            async with self._lock:
                self._waiting = [item for item in self._waiting if item[2] is not waiter]
                heapq.heapify(self._waiting)
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
                if priority <= 2 and not waiter.cancelled()
            )
            historical_waiting = sum(
                1 for priority, _, waiter in self._waiting
                if priority >= 9 and not waiter.cancelled()
            )
            return {
                "active_workloads": self._active,
                "realtime_waiting": realtime_waiting,
                "historical_waiting": historical_waiting,
            }


@dataclass
class TrackedRequest:
    request_id: str
    cancel_event: asyncio.Event
    state: str = "REGISTERED"


class RequestRegistry:
    """Tracks active inference requests and makes cancellation idempotent.

    A repeated request id never creates a second queued/backend request.  Once
    the original request has completed or has been cancelled and cleaned up,
    the caller may safely retry the same deterministic id.
    """

    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._requests: dict[str, TrackedRequest] = {}

    async def register(self, request_id: str) -> TrackedRequest | None:
        async with self._lock:
            if request_id in self._requests:
                return None
            tracked = TrackedRequest(request_id=request_id, cancel_event=asyncio.Event())
            self._requests[request_id] = tracked
            return tracked

    async def set_state(self, request_id: str, state: str) -> None:
        async with self._lock:
            tracked = self._requests.get(request_id)
            if tracked is not None:
                tracked.state = state

    async def cancel(self, request_id: str) -> bool:
        async with self._lock:
            tracked = self._requests.get(request_id)
            if tracked is None:
                # DELETE is deliberately idempotent.  A retry after an already
                # completed cleanup must not be reported as a new failure.
                return False
            tracked.state = "CANCEL_REQUESTED"
            tracked.cancel_event.set()
            return True

    async def unregister(self, request_id: str) -> None:
        async with self._lock:
            self._requests.pop(request_id, None)

    async def snapshot(self) -> dict[str, object]:
        async with self._lock:
            return {
                "tracked_requests": len(self._requests),
                "requests": {
                    request_id: tracked.state
                    for request_id, tracked in sorted(self._requests.items())
                },
            }


class RequestCancelled(Exception):
    pass


async def _wait_for_request_cancel(
    request: Request, cancel_event: asyncio.Event
) -> None:
    while not cancel_event.is_set():
        if await request.is_disconnected():
            cancel_event.set()
            break
        await asyncio.sleep(0.05)


async def _await_cancelable(
    awaitable, cancel_event: asyncio.Event
):
    operation = asyncio.ensure_future(awaitable)
    cancellation = asyncio.create_task(cancel_event.wait())
    try:
        done, _ = await asyncio.wait(
            {operation, cancellation}, return_when=asyncio.FIRST_COMPLETED
        )
        # If acquisition and cancellation become ready in the same loop turn,
        # the acquired resource must be returned to the caller so its acquired
        # flag is set and the single cleanup path can release it.
        if operation in done:
            cancellation.cancel()
            return await operation
        if cancellation in done:
            operation.cancel()
            try:
                await operation
            except BaseException:
                pass
            raise RequestCancelled("inference request was cancelled or disconnected")
        raise RequestCancelled("inference request was cancelled or disconnected")
    except BaseException:
        if not operation.done():
            operation.cancel()
        if not cancellation.done():
            cancellation.cancel()
        raise


def _forward_headers(headers: httpx.Headers) -> dict[str, str]:
    return {key: value for key, value in headers.items() if key.lower() not in HOP_BY_HOP_HEADERS}


async def _acquire_inference_gate(path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_CREAT | os.O_RDWR, 0o660)
    try:
        while True:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_SH | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                await asyncio.sleep(0.05)
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
    memory_provider: Callable[[], tuple[int, int]] = _gpu_memory_snapshot,
) -> FastAPI:
    runtime_config = config or RouterConfig.from_environment()
    runtime_state = RuntimeState(runtime_config.state_path)
    workload_gate = PriorityWorkloadGate(runtime_config.max_concurrent_workloads)
    request_registry = RequestRegistry()
    admission = PersistentGpuAdmission(
        runtime_config.admission_db_path,
        poll_seconds=runtime_config.admission_poll_seconds,
        minimum_free_mb=runtime_config.admission_minimum_free_mb,
        maximum_active=runtime_config.admission_max_active,
        memory_provider=memory_provider,
    )

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
    async def runtime() -> dict[str, object]:
        state = await runtime_state.snapshot()
        state.update(await workload_gate.snapshot())
        state.update(await request_registry.snapshot())
        return state

    @application.get("/_gta/admission")
    async def admission_status() -> dict[str, object]:
        return await admission.snapshot()

    @application.post("/_gta/admission/leases")
    async def acquire_lease(request: LeaseAcquireRequest) -> dict[str, object]:
        try:
            return await admission.acquire(request)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except TimeoutError as exc:
            raise HTTPException(
                status_code=409,
                detail=str(exc),
                headers={"Retry-After": "1"},
            ) from exc

    @application.post("/_gta/admission/leases/{lease_id}/heartbeat")
    async def heartbeat_lease(
        lease_id: str, request: LeaseHeartbeatRequest
    ) -> dict[str, object]:
        try:
            return await admission.heartbeat(lease_id, request)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="lease not found") from exc

    @application.delete("/_gta/admission/leases/{lease_id}")
    async def release_lease(lease_id: str, owner: str) -> dict[str, bool]:
        if not owner:
            raise HTTPException(status_code=422, detail="owner is required")
        released = await admission.release(lease_id, owner)
        if not released:
            raise HTTPException(status_code=404, detail="lease not found")
        return {"released": True}

    @application.delete("/_gta/requests/{request_id}")
    async def cancel_request(request_id: str) -> dict[str, object]:
        if not REQUEST_ID_PATTERN.fullmatch(request_id):
            raise HTTPException(status_code=422, detail="invalid request id")
        cancelled = await request_registry.cancel(request_id)
        return {"request_id": request_id, "cancelled": cancelled}

    @application.api_route(
        "/{path:path}",
        methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS", "HEAD"],
    )
    async def proxy(path: str, request: Request) -> Response:
        upstream_path = "/" + path
        is_workload = request.method in {"POST", "PUT", "PATCH"} and (
            upstream_path in WORKLOAD_PATHS or upstream_path.startswith("/v1/")
        )
        # Consume the ASGI request body before starting a disconnect watcher.
        # Starlette's is_disconnected() reads from the same receive channel;
        # starting it first can steal the body frame and deadlock request.body().
        request_body = await request.body()
        request_id = request.headers.get("x-gta-request-id", "").strip()
        tracked: TrackedRequest | None = None
        if is_workload:
            if not request_id:
                request_id = uuid.uuid4().hex
            if not REQUEST_ID_PATTERN.fullmatch(request_id):
                raise HTTPException(status_code=422, detail="invalid X-GTA-Request-ID")
            tracked = await request_registry.register(request_id)
            if tracked is None:
                raise HTTPException(
                    status_code=409,
                    detail="request id is already active",
                    headers={"Retry-After": "1"},
                )
            await runtime_state.begin()

        inference_gate: int | None = None
        workload_gate_acquired = False
        admission_lease: dict[str, object] | None = None
        admission_owner = "router:" + hashlib.sha256(request_id.encode()).hexdigest()[:32]
        timeout = httpx.Timeout(runtime_config.request_timeout_seconds, connect=2)
        client = httpx.AsyncClient(timeout=timeout, transport=transport)
        disconnect_watcher = (
            asyncio.create_task(_wait_for_request_cancel(request, tracked.cancel_event))
            if is_workload
            else None
        )
        cleaned = False
        cleanup_lock = asyncio.Lock()

        async def cleanup() -> None:
            nonlocal cleaned, inference_gate, admission_lease
            async with cleanup_lock:
                if cleaned:
                    return
                cleaned = True
                if disconnect_watcher is not None:
                    disconnect_watcher.cancel()
                await client.aclose()
                _release_inference_gate(inference_gate)
                inference_gate = None
                if admission_lease is not None:
                    await admission.release(
                        str(admission_lease["lease_id"]), admission_owner
                    )
                    admission_lease = None
                if workload_gate_acquired:
                    await workload_gate.release()
                if is_workload:
                    await runtime_state.end()
                    await request_registry.unregister(request_id)

        try:
            if is_workload:
                priority = workload_gate.parse_priority(request.headers.get("x-gta-priority"))
                await request_registry.set_state(request_id, "WAITING_WORKLOAD")
                await _await_cancelable(
                    workload_gate.acquire(priority), tracked.cancel_event
                )
                workload_gate_acquired = True
                await request_registry.set_state(request_id, "WAITING_ADMISSION")
                workload_class = request.headers.get("x-gta-workload-class") or (
                    workload_gate.workload_class(priority)
                )
                try:
                    admission_lease = await _await_cancelable(
                        admission.acquire(
                            LeaseAcquireRequest(
                                owner=admission_owner,
                                workload_class=workload_class,
                                requested_memory_mb=0,
                                ttl_seconds=min(
                                    7200,
                                    max(60, int(runtime_config.request_timeout_seconds) + 60),
                                ),
                                wait_seconds=runtime_config.request_timeout_seconds,
                            )
                        ),
                        tracked.cancel_event,
                    )
                except (TimeoutError, ValueError) as exc:
                    raise HTTPException(
                        status_code=503,
                        detail=f"GPU admission rejected inference: {exc}",
                        headers={"Retry-After": "1"},
                    ) from exc
                # 共享锁覆盖整个推理响应。训练持有独占锁时, 新请求先登记为 active,
                # 然后等待调度器抢占训练并释放 GPU; 绝不会撞上正在退出的 vLLM。
                await request_registry.set_state(request_id, "WAITING_GPU_LOCK")
                inference_gate = await _await_cancelable(
                    _acquire_inference_gate(runtime_config.gpu_lock_path),
                    tracked.cancel_event,
                )
                await request_registry.set_state(request_id, "WAITING_BACKEND")
                await _await_cancelable(
                    wait_for_backend(client), tracked.cancel_event
                )
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
                content=request_body,
            )
            if is_workload:
                await request_registry.set_state(request_id, "INFERENCE")
                upstream = await _await_cancelable(
                    client.send(upstream_request, stream=True),
                    tracked.cancel_event,
                )
            else:
                upstream = await client.send(upstream_request, stream=True)
        except RequestCancelled as exc:
            await cleanup()
            raise HTTPException(status_code=499, detail=str(exc)) from exc
        except asyncio.CancelledError:
            await cleanup()
            raise
        except HTTPException:
            await cleanup()
            raise
        except httpx.HTTPError as exc:
            await cleanup()
            raise HTTPException(
                status_code=502, detail=f"推理后端连接失败: {type(exc).__name__}"
            ) from exc
        except BaseException:
            await cleanup()
            raise

        async def stream() -> AsyncIterator[bytes]:
            try:
                iterator = upstream.aiter_raw().__aiter__()
                while True:
                    try:
                        if is_workload:
                            chunk = await _await_cancelable(
                                iterator.__anext__(), tracked.cancel_event
                            )
                        else:
                            chunk = await iterator.__anext__()
                    except StopAsyncIteration:
                        break
                    yield chunk
            finally:
                await upstream.aclose()
                await cleanup()

        response_headers = _forward_headers(upstream.headers)
        if is_workload:
            response_headers["x-gta-request-id"] = request_id
        return StreamingResponse(
            stream(),
            status_code=upstream.status_code,
            headers=response_headers,
            media_type=upstream.headers.get("content-type"),
        )

    return application


app = create_router_app()


def main() -> None:
    config = RouterConfig.from_environment()
    uvicorn.run(app, host=config.host, port=config.port, log_level="info")
