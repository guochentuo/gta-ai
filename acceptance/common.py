from __future__ import annotations

import json
import subprocess
import threading
import time
from contextlib import suppress
from pathlib import Path
from typing import Any

import httpx

API_BASE = "http://127.0.0.1:8000"
MODEL = "Qwen/Qwen3.6-27B-FP8"


class GPUMonitor:
    def __init__(self, interval_seconds: float = 0.5) -> None:
        self.interval_seconds = interval_seconds
        self.samples: list[int] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def __enter__(self) -> GPUMonitor:
        self._thread = threading.Thread(target=self._sample, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *_: object) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)

    def _sample(self) -> None:
        while not self._stop.is_set():
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=memory.used",
                    "--format=csv,noheader,nounits",
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            if result.returncode == 0:
                with suppress(IndexError, ValueError):
                    self.samples.append(int(result.stdout.strip().splitlines()[0]))
            self._stop.wait(self.interval_seconds)

    @property
    def peak_memory_mib(self) -> int | None:
        return max(self.samples) if self.samples else None


def post_json(
    path: str, payload: dict[str, Any], *, timeout: float = 900
) -> tuple[dict[str, Any], float]:
    started = time.perf_counter()
    with httpx.Client(timeout=timeout) as client:
        response = client.post(f"{API_BASE}{path}", json=payload)
        response.raise_for_status()
    return response.json(), time.perf_counter() - started


def chat_payload(
    messages: list[dict[str, Any]],
    *,
    max_tokens: int,
    response_schema: dict[str, Any] | None = None,
    temperature: float = 0.0,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": MODEL,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    if response_schema is not None:
        payload["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": "acceptance_result",
                "strict": True,
                "schema": response_schema,
            },
        }
    return payload


def assistant_content(body: dict[str, Any]) -> str:
    content = body["choices"][0]["message"]["content"]
    if not isinstance(content, str):
        raise TypeError("assistant content is not text")
    return content


def parsed_content(body: dict[str, Any]) -> dict[str, Any]:
    value = json.loads(assistant_content(body))
    if not isinstance(value, dict):
        raise TypeError("assistant JSON is not an object")
    return value


def save_result(output_dir: Path, name: str, result: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / f"{name}.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
