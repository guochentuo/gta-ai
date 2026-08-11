# ruff: noqa: RUF001
from __future__ import annotations

import argparse
import json
from pathlib import Path

from common import MODEL, GPUMonitor, post_json, save_result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()

    payload = {
        "model": MODEL,
        "messages": [
            {
                "role": "user",
                "content": (
                    "查询2026-08-07广告系列GTA-42的cost、clicks和conversions，"
                    "必须调用工具，不得编造结果。"
                ),
            }
        ],
        "temperature": 0,
        "max_tokens": 160,
        "chat_template_kwargs": {"enable_thinking": False},
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "get_google_ads_metrics",
                    "description": "获取指定日期和广告系列的Google Ads指标",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "date": {"type": "string"},
                            "campaign_id": {"type": "string"},
                            "metrics": {
                                "type": "array",
                                "items": {
                                    "type": "string",
                                    "enum": ["cost", "clicks", "conversions"],
                                },
                            },
                        },
                        "required": ["date", "campaign_id", "metrics"],
                        "additionalProperties": False,
                    },
                },
            }
        ],
        "tool_choice": "auto",
    }
    with GPUMonitor() as monitor:
        body, elapsed = post_json("/v1/chat/completions", payload, timeout=180)
    choice = body["choices"][0]
    tool_call = choice["message"]["tool_calls"][0]
    arguments = json.loads(tool_call["function"]["arguments"])
    checks = {
        "finish_reason": choice["finish_reason"] == "tool_calls",
        "tool_type": tool_call["type"] == "function",
        "tool_name": tool_call["function"]["name"] == "get_google_ads_metrics",
        "date": arguments["date"] == "2026-08-07",
        "campaign_id": arguments["campaign_id"] == "GTA-42",
        "metrics": set(arguments["metrics"]) == {"cost", "clicks", "conversions"},
        "no_fabricated_content": choice["message"]["content"] is None,
    }
    result = {
        "passed": all(checks.values()),
        "checks": checks,
        "elapsed_seconds": elapsed,
        "peak_gpu_memory_mib": monitor.peak_memory_mib,
        "tool_call": tool_call,
        "parsed_arguments": arguments,
        "usage": body.get("usage"),
    }
    save_result(args.output_dir, "09_tool_call_format", result)
    print(result, flush=True)


if __name__ == "__main__":
    main()
