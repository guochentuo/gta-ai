# ruff: noqa: RUF001
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
from pathlib import Path

import httpx


def build_prompts() -> list[str]:
    destinations = [
        "北京", "上海", "南京", "杭州", "苏州", "无锡", "西安", "成都", "重庆", "广州",
        "深圳", "厦门", "青岛", "长沙", "武汉", "昆明", "大理", "桂林", "三亚", "哈尔滨",
    ]
    templates = [
        "请规划一份{city}三日游，直接给出行程。",
        "带老人去{city}旅行要注意什么？",
        "第一次去{city}，景点和美食应该怎样安排？",
        "{city}什么季节适合旅行？请说明原因。",
    ]
    prompts = [template.format(city=city) for city in destinations for template in templates]
    prompts.extend(
        [
            "2加3等于多少？", "计算125乘以16。", "一小时有多少秒？", "解释什么是百分比。",
            "把‘谢谢你的帮助’翻译成英文。", "把‘Have a nice trip’翻译成中文。",
            "写一句礼貌的酒店入住时间咨询。", "写一封简短的行程确认邮件。",
            "用三句话解释什么是人工智能。", "解释为什么天空看起来是蓝色的。",
            "水在标准大气压下多少摄氏度结冰？", "太阳系中最大的行星是什么？",
            "什么是时差？", "什么是直达航班？", "预算和实际支出有什么区别？",
            "带儿童旅行优先注意什么？", "带老人旅行需要优先注意什么？",
            "夏季户外活动怎样防中暑？", "徒步前需要检查哪些装备？",
            "护照遗失后应该先做什么？", "航班延误后应该怎样处理？",
            "酒店房间与预订不符怎么办？", "旅行中手机没电应该如何应对？",
            "为什么行程中需要留机动时间？", "怎样避免旅行计划安排得太满？",
            "第一次租车需要检查什么？", "购买旅游保险要关注哪些条款？",
            "写一个Python函数，返回列表中的最大值。", "解释HTTP状态码404的含义。",
            "JSON和XML有什么区别？", "数据库索引有什么作用？",
            "请把这句话改得更礼貌：你赶紧回复我。", "写一句简短的生日祝福。",
            "概括健康饮食的三个原则。", "为什么充足睡眠很重要？",
            "如何判断一条信息是否可靠？", "列出做决定前应考虑的三个方面。",
            "用一句话解释什么是取消政策。", "迷路时应该怎样保证安全？",
        ]
    )
    return prompts


async def capture(endpoint: str, model: str, concurrency: int) -> list[dict[str, object]]:
    semaphore = asyncio.Semaphore(concurrency)
    prompts = build_prompts()

    async with httpx.AsyncClient(timeout=300) as client:
        async def generate(index: int, prompt: str) -> tuple[int, dict[str, object]]:
            async with semaphore:
                response = await client.post(
                    endpoint,
                    json={
                        "model": model,
                        "messages": [{
                            "role": "user",
                            "content": prompt + "\n请直接回答，控制在180个汉字以内，不要自我介绍。",
                        }],
                        "temperature": 0.2,
                        "max_tokens": 512,
                        "chat_template_kwargs": {"enable_thinking": False},
                    },
                )
                response.raise_for_status()
                choice = response.json()["choices"][0]
                if choice.get("finish_reason") == "length":
                    raise ValueError(f"truncated preservation answer: {index}")
                answer = choice["message"]["content"].strip()
                if not answer:
                    raise ValueError(f"empty preservation answer: {index}")
                return index, {
                    "messages": [
                        {"role": "user", "content": prompt},
                        {"role": "assistant", "content": answer},
                    ]
                }

        results = await asyncio.gather(
            *(generate(index, prompt) for index, prompt in enumerate(prompts))
        )
    return [record for _, record in sorted(results)]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", default="http://127.0.0.1:18086/v1/chat/completions")
    parser.add_argument("--model", default="Qwen/Qwen3.8-27B-FP8")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--concurrency", type=int, default=5)
    args = parser.parse_args()
    records = asyncio.run(capture(args.endpoint, args.model, args.concurrency))
    encoded = "".join(
        json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
        for record in records
    ).encode()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(encoded)
    manifest = {
        "schema_version": 1,
        "model": args.model,
        "uses_system_prompt": False,
        "record_count": len(records),
        "sha256": hashlib.sha256(encoded).hexdigest(),
    }
    args.manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(manifest, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
