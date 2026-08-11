# ruff: noqa: E501, RUF001
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

from common import GPUMonitor, chat_payload, parsed_content, post_json, save_result


def current_memory_mib() -> int:
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        check=True,
        capture_output=True,
        text=True,
    )
    return int(result.stdout.strip().splitlines()[0])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--requests", type=int, default=20)
    args = parser.parse_args()

    schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "request_id": {"type": "integer"},
            "sum": {"type": "integer"},
            "status": {"type": "string", "enum": ["ok"]},
        },
        "required": ["request_id", "sum", "status"],
        "additionalProperties": False,
    }
    outcomes = []
    baseline = current_memory_mib()
    with GPUMonitor(interval_seconds=0.25) as monitor:
        for request_id in range(1, args.requests + 1):
            left = request_id * 7
            right = request_id * 11
            payload = chat_payload(
                [
                    {
                        "role": "user",
                        "content": f"验收请求{request_id}：计算{left}+{right}，返回请求编号、和以及状态ok。",
                    }
                ],
                max_tokens=80,
                response_schema=schema,
            )
            try:
                body, elapsed = post_json("/v1/chat/completions", payload, timeout=180)
                parsed = parsed_content(body)
                passed = (
                    int(parsed["request_id"]) == request_id
                    and int(parsed["sum"]) == left + right
                    and parsed["status"] == "ok"
                    and body["choices"][0]["finish_reason"] == "stop"
                )
                outcomes.append(
                    {
                        "request_id": request_id,
                        "passed": passed,
                        "elapsed_seconds": elapsed,
                        "memory_after_mib": current_memory_mib(),
                        "parsed": parsed,
                    }
                )
            except Exception as exc:
                outcomes.append(
                    {
                        "request_id": request_id,
                        "passed": False,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )

    result = {
        "passed": all(item["passed"] for item in outcomes),
        "request_count": args.requests,
        "passed_count": sum(1 for item in outcomes if item["passed"]),
        "baseline_memory_mib": baseline,
        "peak_gpu_memory_mib": monitor.peak_memory_mib,
        "final_memory_mib": current_memory_mib(),
        "outcomes": outcomes,
    }
    save_result(args.output_dir, "08_continuous_requests", result)
    print(
        json.dumps({k: v for k, v in result.items() if k != "outcomes"}, ensure_ascii=False),
        flush=True,
    )


if __name__ == "__main__":
    main()
