# ruff: noqa: E501, RUF001
from __future__ import annotations

import argparse
import base64
import mimetypes
from pathlib import Path
from typing import Any

from common import GPUMonitor, chat_payload, parsed_content, post_json, save_result


def image_part(path: Path) -> dict[str, Any]:
    mime = mimetypes.guess_type(path.name)[0] or "image/jpeg"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return {
        "type": "image_url",
        "image_url": {"url": f"data:{mime};base64,{encoded}"},
    }


def run_visual_test(
    output_dir: Path,
    name: str,
    content: list[dict[str, Any]],
    schema: dict[str, Any],
    validator: Any,
) -> dict[str, Any]:
    messages = [{"role": "user", "content": content}]
    with GPUMonitor() as monitor:
        body, elapsed = post_json(
            "/v1/chat/completions",
            chat_payload(messages, max_tokens=500, response_schema=schema),
        )
    parsed = parsed_content(body)
    checks = validator(parsed)
    result = {
        "passed": all(checks.values()),
        "checks": checks,
        "elapsed_seconds": elapsed,
        "peak_gpu_memory_mib": monitor.peak_memory_mib,
        "parsed": parsed,
        "usage": body.get("usage"),
    }
    save_result(output_dir, name, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("media_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()

    ad_schema = {
        "type": "object",
        "properties": {
            "brand": {"type": "string"},
            "claim_percent": {"type": "number"},
            "trial_days": {"type": "integer"},
            "cta": {"type": "string"},
            "pilot_size": {"type": "integer"},
            "risk": {"type": "string"},
        },
        "required": ["brand", "claim_percent", "trial_days", "cta", "pilot_size", "risk"],
        "additionalProperties": False,
    }
    ad = run_visual_test(
        args.output_dir,
        "04_ad_creative",
        [
            image_part(args.media_dir / "ad-creative.png"),
            {
                "type": "text",
                "text": "读取广告素材中的品牌、效果主张、试用期、CTA和免责声明样本量，并指出审核风险。",
            },
        ],
        ad_schema,
        lambda p: {
            "brand": "GTA" in str(p["brand"]).upper(),
            "claim_percent": abs(float(p["claim_percent"]) - 30) < 0.1,
            "trial_days": int(p["trial_days"]) == 14,
            "cta": "START" in str(p["cta"]).upper(),
            "pilot_size": int(p["pilot_size"]) == 12,
            "risk_present": len(str(p["risk"]).strip()) >= 5,
        },
    )

    web_schema = {
        "type": "object",
        "properties": {
            "headline": {"type": "string"},
            "sessions": {"type": "integer"},
            "session_change_percent": {"type": "number"},
            "conversions": {"type": "integer"},
            "conversion_change_percent": {"type": "number"},
            "current_cvr_percent": {"type": "number"},
            "previous_cvr_percent": {"type": "number"},
            "tracking_gap": {"type": "string"},
            "recommended_action": {"type": "string"},
        },
        "required": [
            "headline",
            "sessions",
            "session_change_percent",
            "conversions",
            "conversion_change_percent",
            "current_cvr_percent",
            "previous_cvr_percent",
            "tracking_gap",
            "recommended_action",
        ],
        "additionalProperties": False,
    }
    web = run_visual_test(
        args.output_dir,
        "05_webpage_screenshot",
        [
            image_part(args.media_dir / "webpage-screenshot.png"),
            {
                "type": "text",
                "text": "读取这个网页截图的指标和告警，并给出数据分析前最应执行的一项动作。百分比用数值20、-10、3、4表示。",
            },
        ],
        web_schema,
        lambda p: {
            "sessions": int(p["sessions"]) == 120000,
            "session_change": abs(float(p["session_change_percent"]) - 20) < 0.1,
            "conversions": int(p["conversions"]) == 3600,
            "conversion_change": abs(float(p["conversion_change_percent"]) + 10) < 0.1,
            "current_cvr": abs(float(p["current_cvr_percent"]) - 3) < 0.1,
            "previous_cvr": abs(float(p["previous_cvr_percent"]) - 4) < 0.1,
            "tracking_gap": "0200" in str(p["tracking_gap"]).replace(":", "")
            and "0400" in str(p["tracking_gap"]).replace(":", ""),
            "action_present": len(str(p["recommended_action"]).strip()) >= 5,
        },
    )

    scene_schema = {
        "type": "object",
        "properties": {
            "product": {"type": "string"},
            "minutes_saved": {"type": "integer"},
            "monthly_price_usd": {"type": "integer"},
            "cta": {"type": "string"},
            "ordered_scenes": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["product", "minutes_saved", "monthly_price_usd", "cta", "ordered_scenes"],
        "additionalProperties": False,
    }
    frames = sorted((args.media_dir / "keyframes").glob("frame-*.jpg"))
    frame_content = [image_part(path) for path in frames]
    frame_content.append(
        {
            "type": "text",
            "text": "这些是同一视频按时间顺序抽取的关键帧。还原产品、节省时间、月价、CTA和四幕顺序。",
        }
    )
    video = run_visual_test(
        args.output_dir,
        "06_video_keyframes",
        frame_content,
        scene_schema,
        lambda p: {
            "four_frames_loaded": len(frames) == 4,
            "product": "ATLAS" in str(p["product"]).upper(),
            "minutes": int(p["minutes_saved"]) == 30,
            "price": int(p["monthly_price_usd"]) == 29,
            "cta": "TRIAL" in str(p["cta"]).upper(),
            "scene_count": len(p["ordered_scenes"]) == 4,
        },
    )

    tests = {
        "ad_creative": ad["passed"],
        "webpage_screenshot": web["passed"],
        "video_keyframes": video["passed"],
    }
    save_result(args.output_dir, "vision_summary", {"passed": all(tests.values()), "tests": tests})
    print(tests, flush=True)


if __name__ == "__main__":
    main()
