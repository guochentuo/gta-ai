"""GPU状态采集与跨进程准入服务。"""

from __future__ import annotations

import asyncio
import re
import sqlite3
import subprocess
import time
import uuid
from collections.abc import Callable
from pathlib import Path

from pydantic import BaseModel, Field

METRIC_PATTERN = re.compile(
    r"^(?P<name>[a-zA-Z_:][a-zA-Z0-9_:]*)(?:\{[^}]*\})?\s+(?P<value>[-+0-9.eE]+)$"
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


def _gpu_summary() -> dict[str, str | int | float]:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total,memory.used,memory.free,"
                "utilization.gpu,temperature.gpu,power.draw,power.limit",
                "--format=csv,noheader,nounits",
                "--id=0",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=2,
        )
        name, driver, total, used, free, utilization, temperature, power, power_limit = (
            value.strip() for value in result.stdout.strip().splitlines()[0].split(",")
        )
        return {
            "gpu": name,
            "driver": driver,
            "memory_total_mb": int(total),
            "memory_used_mb": int(used),
            "memory_free_mb": int(free),
            "gpu_utilization_percent": int(utilization),
            "temperature_celsius": int(temperature),
            "power_watts": round(float(power), 1),
            "power_limit_watts": round(float(power_limit), 1),
        }
    except (OSError, ValueError, IndexError, subprocess.SubprocessError):
        return {"gpu": "无法读取"}


def _metric_values(payload: str) -> dict[str, float]:
    values: dict[str, float] = {}
    for line in payload.splitlines():
        match = METRIC_PATTERN.fullmatch(line.strip())
        if match is None:
            continue
        name = match.group("name")
        values[name] = values.get(name, 0.0) + float(match.group("value"))
    return values


def _memory_gb(gpu: dict[str, str | int | float]) -> str:
    mib_to_gb = 1024 * 1024 / 1_000_000_000
    used = int(gpu.get("memory_used_mb", 0)) * mib_to_gb
    total = int(gpu.get("memory_total_mb", 0)) * mib_to_gb
    return f"{used:.1f}/{total:.1f} GB"


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
                    heartbeat_at REAL,
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

    def _class_limit(self, workload_class: str) -> int:
        if workload_class in {"REALTIME_UNDERSTANDING", "HISTORICAL_UNDERSTANDING"}:
            return 1
        if workload_class in {
            "REALTIME_PLAYBACK",
            "REALTIME_FEATURE",
            "HISTORICAL_FEATURE",
        }:
            return self._maximum_active
        return 1

    @staticmethod
    def _purge(connection: sqlite3.Connection, now: float) -> None:
        stale_before = now - 15.0
        connection.execute(
            "DELETE FROM gpu_lease WHERE expires_at <= ? OR heartbeat_at <= ?",
            (now, stale_before),
        )
        connection.execute(
            "DELETE FROM gpu_waiter WHERE expires_at <= ? "
            "OR heartbeat_at IS NULL OR heartbeat_at <= ?",
            (now, stale_before),
        )

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
                    "INSERT INTO gpu_waiter "
                    "(waiter_id, owner, workload_class, priority, requested_memory_mb, "
                    "created_at, heartbeat_at, expires_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        waiter_id,
                        request.owner,
                        workload_class,
                        priority,
                        request.requested_memory_mb,
                        now,
                        now,
                        now + 15.0,
                    ),
                )
        try:
            while True:
                async with self._lock:
                    now = time.time()
                    with self._connect() as connection:
                        self._purge(connection, now)
                        connection.execute(
                            "UPDATE gpu_waiter SET heartbeat_at = ?, expires_at = ? "
                            "WHERE waiter_id = ?",
                            (now, now + 15.0, waiter_id),
                        )
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

    async def heartbeat(self, lease_id: str, request: LeaseHeartbeatRequest) -> dict[str, object]:
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
                leases = [
                    dict(row)
                    for row in connection.execute(
                        "SELECT * FROM gpu_lease ORDER BY priority, acquired_at"
                    )
                ]
                waiters = [
                    dict(row)
                    for row in connection.execute(
                        "SELECT * FROM gpu_waiter ORDER BY priority, created_at"
                    )
                ]
            used_mb, free_mb = self._memory_provider()
            return {
                "schema_version": 1,
                "gpu_memory_used_mb": used_mb,
                "gpu_memory_free_mb": free_mb,
                "minimum_free_mb": self._minimum_free_mb,
                "leases": leases,
                "waiters": waiters,
            }

