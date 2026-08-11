# ruff: noqa: E501, RUF001
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from common import (
    GPUMonitor,
    assistant_content,
    chat_payload,
    parsed_content,
    post_json,
    save_result,
)


def close(actual: float, expected: float, tolerance: float = 0.015) -> bool:
    return abs(actual - expected) <= tolerance


def run_chinese_analysis(output_dir: Path) -> dict[str, Any]:
    schema = {
        "type": "object",
        "properties": {
            "previous_ctr": {"type": "number"},
            "current_ctr": {"type": "number"},
            "previous_cvr": {"type": "number"},
            "current_cvr": {"type": "number"},
            "previous_cpa": {"type": "number"},
            "current_cpa": {"type": "number"},
            "previous_roas": {"type": "number"},
            "current_roas": {"type": "number"},
            "cpa_relative_change": {"type": "number"},
            "roas_relative_change": {"type": "number"},
            "diagnosis": {"type": "string"},
            "recommended_action": {"type": "string"},
        },
        "required": [
            "previous_ctr",
            "current_ctr",
            "previous_cvr",
            "current_cvr",
            "previous_cpa",
            "current_cpa",
            "previous_roas",
            "current_roas",
            "cpa_relative_change",
            "roas_relative_change",
            "diagnosis",
            "recommended_action",
        ],
        "additionalProperties": False,
    }
    messages = [
        {
            "role": "system",
            "content": (
                "你是严谨的广告数据分析师。比率使用小数表示，例如4%=0.04；"
                "相对变化使用小数表示。只依据输入数据，不虚构原因。"
            ),
        },
        {
            "role": "user",
            "content": (
                "上一周期：展示80000，点击3200，转化160，花费24000元，收入64000元。\n"
                "当前周期：展示100000，点击3500，转化140，花费35000元，收入56000元。\n"
                "计算CTR、CVR、CPA、ROAS及CPA与ROAS的相对变化，并给出谨慎诊断和下一步动作。"
            ),
        },
    ]
    with GPUMonitor() as monitor:
        body, elapsed = post_json(
            "/v1/chat/completions",
            chat_payload(messages, max_tokens=420, response_schema=schema),
        )
    parsed = parsed_content(body)
    checks = {
        "previous_ctr": close(float(parsed["previous_ctr"]), 0.04),
        "current_ctr": close(float(parsed["current_ctr"]), 0.035),
        "previous_cvr": close(float(parsed["previous_cvr"]), 0.05),
        "current_cvr": close(float(parsed["current_cvr"]), 0.04),
        "previous_cpa": close(float(parsed["previous_cpa"]), 150, 0.1),
        "current_cpa": close(float(parsed["current_cpa"]), 250, 0.1),
        "previous_roas": close(float(parsed["previous_roas"]), 2.6667),
        "current_roas": close(float(parsed["current_roas"]), 1.6),
        "cpa_relative_change": close(float(parsed["cpa_relative_change"]), 2 / 3),
        "roas_relative_change": close(float(parsed["roas_relative_change"]), -0.4),
        "diagnosis_present": len(str(parsed["diagnosis"]).strip()) >= 10,
        "action_present": len(str(parsed["recommended_action"]).strip()) >= 10,
    }
    result = {
        "passed": all(checks.values()),
        "checks": checks,
        "elapsed_seconds": elapsed,
        "peak_gpu_memory_mib": monitor.peak_memory_mib,
        "parsed": parsed,
        "usage": body.get("usage"),
    }
    save_result(output_dir, "01_chinese_analysis", result)
    return result


def build_long_article() -> str:
    background = (
        "团队在同一观察窗口内同步调整了落地页、受众、出价与素材。报告应区分事实、"
        "假设和建议，并说明样本规模、归因窗口与数据缺口。"
    )
    sections = [f"第{i}节 背景说明：{background}{background}" for i in range(1, 15)]
    seeded = [
        "标题：新方案让广告ROAS提升50%，已经得到完全证明。",
        "数据段：旧ROAS为4.0，新ROAS为4.4，因此提升幅度是50%。",
        "因果段：上线蓝色按钮的同一天转化率上升，所以可以证明蓝色按钮是唯一原因。",
        "承诺段：购买本服务，保证每位客户30天内收入翻倍，绝无例外。",
        "隐私段：为了提高匹配率，我们会在未取得额外授权时上传手机号和购买记录。",
        "时效段：本文标注为2026年最新结论，但引用的行业基准来自2021年且未注明来源。",
    ]
    return "\n\n".join([sections[0], *seeded, *sections[1:]])


def run_article_audit(output_dir: Path) -> dict[str, Any]:
    codes = [
        "NUMERIC_CONTRADICTION",
        "CAUSALITY_OVERCLAIM",
        "GUARANTEE_CLAIM",
        "PRIVACY_RISK",
        "OUTDATED_OR_UNSOURCED",
        "OTHER",
    ]
    issue_schema = {
        "type": "object",
        "properties": {
            "code": {"type": "string", "enum": codes},
            "severity": {"type": "string", "enum": ["low", "medium", "high"]},
            "evidence": {"type": "string"},
            "explanation": {"type": "string"},
            "recommended_fix": {"type": "string"},
        },
        "required": ["code", "severity", "evidence", "explanation", "recommended_fix"],
        "additionalProperties": False,
    }
    schema = {
        "type": "object",
        "properties": {
            "risk_level": {"type": "string", "enum": ["low", "medium", "high"]},
            "publish_decision": {"type": "string", "enum": ["approve", "revise", "reject"]},
            "issues": {"type": "array", "items": issue_schema},
            "summary": {"type": "string"},
        },
        "required": ["risk_level", "publish_decision", "issues", "summary"],
        "additionalProperties": False,
    }
    messages = [
        {
            "role": "system",
            "content": "审核长文章中的数字矛盾、因果夸大、保证性承诺、隐私风险和过时或无来源信息。证据必须来自原文。",
        },
        {"role": "user", "content": f"请审核以下待发布文章：\n\n{build_long_article()}"},
    ]
    with GPUMonitor() as monitor:
        body, elapsed = post_json(
            "/v1/chat/completions",
            chat_payload(messages, max_tokens=900, response_schema=schema),
        )
    parsed = parsed_content(body)
    found = {str(issue["code"]) for issue in parsed["issues"]}
    required = {
        "NUMERIC_CONTRADICTION",
        "CAUSALITY_OVERCLAIM",
        "GUARANTEE_CLAIM",
        "PRIVACY_RISK",
        "OUTDATED_OR_UNSOURCED",
    }
    checks = {
        "all_seeded_issue_types_found": required <= found,
        "decision_blocks_publish": parsed["publish_decision"] in {"revise", "reject"},
        "risk_is_high": parsed["risk_level"] == "high",
        "json_issue_count": len(parsed["issues"]) >= 5,
    }
    result = {
        "passed": all(checks.values()),
        "checks": checks,
        "found_codes": sorted(found),
        "elapsed_seconds": elapsed,
        "peak_gpu_memory_mib": monitor.peak_memory_mib,
        "article_characters": len(build_long_article()),
        "parsed": parsed,
        "usage": body.get("usage"),
    }
    save_result(output_dir, "02_long_article_audit", result)
    return result


def run_article_generation(output_dir: Path) -> dict[str, Any]:
    messages = [
        {
            "role": "system",
            "content": (
                "你是企业内容编辑。只能使用简报中的事实，不能编造客户、价格、认证、排名或效果保证。"
                "文章须包含标题、导语、三个小节和结尾行动号召。"
            ),
        },
        {
            "role": "user",
            "content": (
                "为内部产品 Atlas Insights 写一篇450至650个中文字符的介绍文章。事实简报："
                "它用于导入CSV、发现指标异常、导出严格JSON；内部测试有12名分析师参与；"
                "报告制作时间中位数从90分钟降至55分钟；目前没有公开价格；"
                "不能承诺任何业务结果；行动号召是申请内部试用。"
            ),
        },
    ]
    with GPUMonitor() as monitor:
        body, elapsed = post_json(
            "/v1/chat/completions",
            chat_payload(messages, max_tokens=850, temperature=0.4),
        )
    content = assistant_content(body)
    required_terms = ["Atlas Insights", "CSV", "JSON", "12", "90", "55", "申请", "试用"]
    forbidden_terms = ["价格为", "保证增长", "行业第一", "获得认证"]
    checks = {
        "all_brief_facts_present": all(term in content for term in required_terms),
        "no_forbidden_fabrication": not any(term in content for term in forbidden_terms),
        "reasonable_length": 350 <= len(content) <= 1100,
        "has_structure": content.count("\n") >= 4,
    }
    result = {
        "passed": all(checks.values()),
        "checks": checks,
        "elapsed_seconds": elapsed,
        "peak_gpu_memory_mib": monitor.peak_memory_mib,
        "output_characters": len(content),
        "content": content,
        "usage": body.get("usage"),
    }
    save_result(output_dir, "03_article_generation", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    results = {
        "chinese_analysis": run_chinese_analysis(args.output_dir),
        "long_article_audit": run_article_audit(args.output_dir),
        "article_generation": run_article_generation(args.output_dir),
    }
    summary = {name: result["passed"] for name, result in results.items()}
    save_result(
        args.output_dir, "content_summary", {"passed": all(summary.values()), "tests": summary}
    )
    print(summary, flush=True)


if __name__ == "__main__":
    main()
