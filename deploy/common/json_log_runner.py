#!/usr/bin/env python3
"""Run one service and persist its stdout/stderr as bounded JSON lines."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import signal
import subprocess
import sys
from pathlib import Path


def timestamp() -> str:
    return dt.datetime.now(dt.UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def infer_level(message: str) -> str:
    lowered = message.lower()
    if "error" in lowered or "exception" in lowered or "traceback" in lowered:
        return "ERROR"
    if "warn" in lowered:
        return "WARN"
    return "INFO"


class JsonLog:
    def __init__(self, path: Path, service: str, module: str, max_bytes: int, backups: int):
        self.path = path
        self.service = service
        self.module = module
        self.max_bytes = max_bytes
        self.backups = backups
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o750)

    def rotate(self, incoming: int) -> None:
        current = self.path.stat().st_size if self.path.exists() else 0
        if current + incoming <= self.max_bytes:
            return
        oldest = self.path.with_name(f"{self.path.name}.{self.backups}")
        if oldest.exists():
            oldest.unlink()
        for index in range(self.backups - 1, 0, -1):
            source = self.path.with_name(f"{self.path.name}.{index}")
            if source.exists():
                source.replace(self.path.with_name(f"{self.path.name}.{index + 1}"))
        if self.path.exists():
            self.path.replace(self.path.with_name(f"{self.path.name}.1"))

    def write(self, event: str, message: str, level: str | None = None, **fields: object) -> None:
        payload = {
            "timestamp": timestamp(),
            "level": level or infer_level(message),
            "service": self.service,
            "module": self.module,
            "event": event,
            "pid": os.getpid(),
            "message": message.rstrip("\r\n"),
            **fields,
        }
        encoded = (json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n").encode()
        self.rotate(len(encoded))
        with self.path.open("ab") as output:
            output.write(encoded)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--service", required=True)
    parser.add_argument("--module", required=True)
    parser.add_argument("--log-file", required=True, type=Path)
    parser.add_argument("--max-size-mb", type=int, default=20)
    parser.add_argument("--backups", type=int, default=10)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command and args.command[0] == "--" else args.command
    if not command:
        parser.error("missing command after --")

    log = JsonLog(
        args.log_file,
        args.service,
        args.module,
        args.max_size_mb * 1024 * 1024,
        args.backups,
    )
    child = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )

    def forward(signum: int, _frame: object) -> None:
        if child.poll() is None:
            child.send_signal(signum)

    signal.signal(signal.SIGTERM, forward)
    signal.signal(signal.SIGINT, forward)
    log.write("service_started", "service process started", child_pid=child.pid)
    assert child.stdout is not None
    for line in child.stdout:
        log.write("service_output", line)
    exit_code = child.wait()
    log.write(
        "service_stopped",
        "service process stopped",
        level="INFO" if exit_code == 0 else "ERROR",
        child_pid=child.pid,
        exit_code=exit_code,
    )
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
