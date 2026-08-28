from __future__ import annotations

import asyncio
import heapq
import json
import time
from dataclasses import dataclass
from pathlib import Path


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
    def __init__(self, maximum_active: int) -> None:
        self._maximum_active = max(1, maximum_active)
        self._active = 0
        self._sequence = 0
        self._waiting: list[tuple[int, int, asyncio.Future[None]]] = []
        self._lock = asyncio.Lock()

    @staticmethod
    def parse_priority(value: str | None) -> int:
        normalized = (value or "P1").strip().upper()
        return {"P0": 0, "P1": 1, "P2": 2, "P9": 9, "P10": 10}.get(normalized, 1)

    @staticmethod
    def workload_class(priority: int) -> str:
        return {
            0: "REALTIME_PLAYBACK",
            1: "REALTIME_FEATURE",
            2: "REALTIME_UNDERSTANDING",
            9: "HISTORICAL_FEATURE",
            10: "HISTORICAL_UNDERSTANDING",
        }.get(priority, "REALTIME_FEATURE")

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
            return {
                "active_workloads": self._active,
                "realtime_waiting": sum(
                    1 for priority, _, waiter in self._waiting
                    if priority <= 2 and not waiter.cancelled()
                ),
                "historical_waiting": sum(
                    1 for priority, _, waiter in self._waiting
                    if priority >= 9 and not waiter.cancelled()
                ),
            }


@dataclass
class TrackedRequest:
    request_id: str
    cancel_event: asyncio.Event
    state: str = "REGISTERED"


class RequestRegistry:
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
