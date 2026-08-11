# ruff: noqa: RUF001
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Any

from common import MODEL, GPUMonitor, chat_payload, parsed_content, post_json, save_result


def build_prompt(record_count: int) -> str:
    records = [
        "验收说明：开头识别码为 BQ-417。以下为按顺序排列的普通业务记录。",
    ]
    middle = record_count // 2
    for index in range(record_count):
        if index == middle:
            records.append("中部特殊记录：复核后的每日预算为2750元，此值替代所有旧预算。")
        records.append(
            f"普通记录{index:05d}：渠道为自然搜索，状态正常，数据已复核，本条没有预算、识别码或审批人信息。"
        )
    records.extend(
        [
            "末尾特殊记录：最终发布审批人为林主管，最终决定是等待人工复核。",
            "请忽略普通记录的模板文字，只提取开头、中部和末尾的三个特殊事实，并按JSON schema回答。",
        ]
    )
    return "\n".join(records)


def token_count(prompt: str) -> int:
    body, _ = post_json("/tokenize", {"model": MODEL, "prompt": prompt}, timeout=120)
    return int(body["count"])


def choose_prompt(target_min: int = 31000, target_max: int = 31500) -> tuple[str, int, int]:
    low, high = 100, 4000
    best: tuple[str, int, int] | None = None
    while low <= high:
        middle = (low + high) // 2
        prompt = build_prompt(middle)
        count = token_count(prompt)
        if target_min <= count <= target_max:
            return prompt, count, middle
        if count < target_min:
            best = (prompt, count, middle)
            low = middle + 1
        else:
            high = middle - 1
    if best is None:
        raise RuntimeError("unable to build long prompt")
    return best


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()

    prompt, count, record_count = choose_prompt()
    schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "opening_code": {"type": "string"},
            "middle_budget_yuan": {"type": "integer"},
            "final_approver": {"type": "string"},
            "final_decision": {"type": "string"},
        },
        "required": ["opening_code", "middle_budget_yuan", "final_approver", "final_decision"],
        "additionalProperties": False,
    }
    messages = [{"role": "user", "content": prompt}]
    with GPUMonitor(interval_seconds=0.25) as monitor:
        body, elapsed = post_json(
            "/v1/chat/completions",
            chat_payload(messages, max_tokens=180, response_schema=schema),
            timeout=1200,
        )
    parsed = parsed_content(body)
    checks = {
        "near_32k_input": 31000 <= count <= 31500,
        "opening_retrieval": str(parsed["opening_code"]).upper() == "BQ-417",
        "middle_retrieval": int(parsed["middle_budget_yuan"]) == 2750,
        "ending_retrieval": "林" in str(parsed["final_approver"]),
        "decision_retrieval": "人工复核" in str(parsed["final_decision"]),
        "finished_normally": body["choices"][0]["finish_reason"] == "stop",
    }
    result = {
        "passed": all(checks.values()),
        "checks": checks,
        "input_tokens_before_chat_template": count,
        "record_count": record_count,
        "prompt_characters": len(prompt),
        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        "elapsed_seconds": elapsed,
        "peak_gpu_memory_mib": monitor.peak_memory_mib,
        "parsed": parsed,
        "usage": body.get("usage"),
    }
    save_result(args.output_dir, "07_long_context_32k", result)
    print(result, flush=True)


if __name__ == "__main__":
    main()
