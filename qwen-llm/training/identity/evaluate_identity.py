from __future__ import annotations

# ruff: noqa: RUF003
import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import httpx


def _contains_all(text: str, values: list[str]) -> bool:
    return all(value.lower() in text.lower() for value in values)


def _contains_any(text: str, values: list[str]) -> bool:
    return not values or any(value.lower() in text.lower() for value in values)


def score_text(text: str, case: dict[str, Any]) -> dict[str, Any]:
    required = case.get("required", [])
    required_any = case.get("required_any", [])
    required_any_groups = case.get("required_any_groups", [])
    forbidden = case.get("forbidden", [])
    passed = (
        _contains_all(text, required)
        and _contains_any(text, required_any)
        and all(_contains_any(text, group) for group in required_any_groups)
        and not any(value.lower() in text.lower() for value in forbidden)
    )
    return {
        "passed": passed,
        "missing": [value for value in required if value.lower() not in text.lower()],
        "missing_any": (
            required_any if required_any and not _contains_any(text, required_any) else []
        ),
        "missing_any_groups": [
            group for group in required_any_groups if not _contains_any(text, group)
        ],
        "forbidden_hits": [value for value in forbidden if value.lower() in text.lower()],
    }


def evaluate_case(
    client: httpx.Client, endpoint: str, model: str, case: dict[str, Any]
) -> dict[str, Any]:
    started = time.monotonic()
    response = client.post(
        endpoint,
        json={
            "model": model,
            # 验收禁止 system 消息，防止用上下文伪装成权重微调结果。
            "messages": [{"role": "user", "content": case["prompt"]}],
            "temperature": 0,
            "max_tokens": 400,
            "chat_template_kwargs": {"enable_thinking": False},
        },
    )
    response.raise_for_status()
    text = response.json()["choices"][0]["message"]["content"]
    return {
        "id": case["id"],
        "elapsed_ms": round((time.monotonic() - started) * 1000),
        "response": text,
        **score_text(text, case),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=300)
    args = parser.parse_args()

    definitions = json.loads(args.cases.read_text(encoding="utf-8"))
    results: list[dict[str, Any]] = []
    with httpx.Client(timeout=args.timeout) as client:
        for case in definitions["hard_cases"] + definitions.get("smoke_cases", []):
            result = evaluate_case(client, args.endpoint, args.model, case)
            results.append(result)
            print(f"{result['id']}: {'PASS' if result['passed'] else 'FAIL'}")
    hard_ids = {case["id"] for case in definitions["hard_cases"]}
    hard_passed = all(result["passed"] for result in results if result["id"] in hard_ids)
    smoke_passed = all(result["passed"] for result in results if result["id"] not in hard_ids)
    report = {
        "schema_version": 1,
        "uses_system_prompt": False,
        "model": args.model,
        "hard_passed": hard_passed,
        "smoke_passed": smoke_passed,
        "passed": hard_passed and smoke_passed,
        "results": results,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
