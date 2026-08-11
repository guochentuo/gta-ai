from __future__ import annotations

import argparse
import subprocess
import time
from pathlib import Path

import httpx
from common import GPUMonitor, save_result


def command(*args: str, check: bool = True) -> str:
    result = subprocess.run(args, check=check, capture_output=True, text=True)
    return result.stdout.strip()


def user_systemctl(*args: str) -> str:
    return command("systemctl", "--user", *args)


def gpu_memory_mib() -> int:
    value = command(
        "nvidia-smi",
        "--query-gpu=memory.used",
        "--format=csv,noheader,nounits",
    )
    return int(value.splitlines()[0])


def service_property(service: str, name: str, *, user: bool = False) -> str:
    prefix = ["systemctl"]
    if user:
        prefix.append("--user")
    return command(*prefix, "show", service, f"--property={name}", "--value")


def api_ready() -> bool:
    try:
        response = httpx.get("http://127.0.0.1:8000/v1/models", timeout=2)
        response.raise_for_status()
        model = response.json()["data"][0]
        return model["id"] == "Qwen/Qwen3.6-27B-FP8" and model["max_model_len"] == 32768
    except (httpx.HTTPError, KeyError, IndexError, TypeError, ValueError):
        return False


def wait_ready(label: str, timeout_seconds: int = 600) -> float:
    started = time.perf_counter()
    next_report = 0
    while time.perf_counter() - started < timeout_seconds:
        elapsed = time.perf_counter() - started
        if api_ready():
            print(f"{label}: ready after {elapsed:.2f}s", flush=True)
            return elapsed
        if elapsed >= next_report:
            print(f"{label}: waiting {elapsed:.0f}s", flush=True)
            next_report += 10
        time.sleep(2)
    raise TimeoutError(f"{label} did not become ready within {timeout_seconds}s")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()

    worker_pid_before = service_property("gta-worker.service", "MainPID")
    worker_restarts_before = service_property("gta-worker.service", "NRestarts")
    phases: dict[str, object] = {}

    try:
        with GPUMonitor(interval_seconds=0.5) as monitor:
            stop_started = time.perf_counter()
            user_systemctl("stop", "gta-ai-vllm.service")
            stop_seconds = time.perf_counter() - stop_started
            stopped_memory = gpu_memory_mib()
            phases["stop"] = {
                "elapsed_seconds": stop_seconds,
                "service_state": service_property("gta-ai-vllm.service", "ActiveState", user=True),
                "api_unavailable": not api_ready(),
                "gpu_memory_mib": stopped_memory,
            }
            print(f"stop: completed after {stop_seconds:.2f}s", flush=True)

            user_systemctl("start", "gta-ai-vllm.service")
            start_seconds = wait_ready("start")
            start_args = command("podman", "inspect", "gta-ai-vllm", "--format", "{{json .Args}}")
            phases["start"] = {
                "ready_seconds": start_seconds,
                "service_state": service_property("gta-ai-vllm.service", "ActiveState", user=True),
                "gpu_memory_mib": gpu_memory_mib(),
                "persistent_script_without_random_kv_calibration": (
                    "calculate-kv-scales" not in start_args
                ),
            }

            restart_started = time.perf_counter()
            user_systemctl("restart", "gta-ai-vllm.service")
            restart_control_seconds = time.perf_counter() - restart_started
            restart_ready_seconds = wait_ready("restart")
            phases["restart"] = {
                "control_seconds": restart_control_seconds,
                "ready_seconds_after_control": restart_ready_seconds,
                "total_seconds": restart_control_seconds + restart_ready_seconds,
                "service_state": service_property("gta-ai-vllm.service", "ActiveState", user=True),
                "gpu_memory_mib": gpu_memory_mib(),
            }

        worker_pid_after = service_property("gta-worker.service", "MainPID")
        worker_restarts_after = service_property("gta-worker.service", "NRestarts")
        checks = {
            "stop_inactive": phases["stop"]["service_state"] == "inactive",  # type: ignore[index]
            "stop_api_unavailable": phases["stop"]["api_unavailable"] is True,  # type: ignore[index]
            "gpu_released": phases["stop"]["gpu_memory_mib"] < 1024,  # type: ignore[index,operator]
            "start_ready": phases["start"]["service_state"] == "active",  # type: ignore[index]
            "restart_ready": phases["restart"]["service_state"] == "active",  # type: ignore[index]
            "peak_below_42gb": (monitor.peak_memory_mib or 999999) < 43008,
            "worker_pid_stable": worker_pid_before == worker_pid_after,
            "worker_no_restart": worker_restarts_before == worker_restarts_after,
            "autostart_disabled": (
                command(
                    "systemctl",
                    "--user",
                    "is-enabled",
                    "gta-ai-vllm.service",
                    check=False,
                )
                == "disabled"
            ),
        }
        result = {
            "passed": all(checks.values()),
            "checks": checks,
            "phases": phases,
            "peak_gpu_memory_mib": monitor.peak_memory_mib,
            "gta_worker_main_pid_before": worker_pid_before,
            "gta_worker_main_pid_after": worker_pid_after,
            "gta_worker_restarts_before": worker_restarts_before,
            "gta_worker_restarts_after": worker_restarts_after,
        }
        save_result(args.output_dir, "10_service_lifecycle", result)
        print(result, flush=True)
    finally:
        if not api_ready():
            user_systemctl("start", "gta-ai-vllm.service")


if __name__ == "__main__":
    main()
