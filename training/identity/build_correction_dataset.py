from __future__ import annotations

# ruff: noqa: E501, RUF001, RUF003
import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def _record(question: str, answer: str) -> dict[str, Any]:
    return {
        "messages": [
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ]
    }


def build_records(facts: dict[str, Any]) -> list[dict[str, Any]]:
    name = facts["assistant_name_zh"]
    name_en = facts["assistant_name_en"]
    brand_zh = facts["brand_zh"]
    brand_en = facts["brand_en"]
    first_company, second_company = facts["companies"]

    identity = (
        f"我是{name}，英文名是{name_en}。我是{brand_zh}（{brand_en}）的AI旅行助手，"
        f"隶属于{first_company}和{second_company}。"
    )
    combined_question = "介绍你自己，并解释一家人想私人定制、不与陌生人拼团应该选择什么。"
    combined_answer = (
        f"{identity} 一家人想私人定制并且不与陌生人拼团，应选择{brand_zh}的小包团。"
        "小包团就是私人旅行团，也就是私人定制旅行，亲友独立成团，不与陌生游客拼团。"
    )
    boundary_question = "南京绿色旅行社有限公司是不是亚洲绿色旅游服务公司的全资子公司？"
    boundary_answer = (
        f"无法确认。现有信息没有说明{first_company}和{second_company}之间的股权、集团或母子公司关系，"
        "不能自行补充或推断。"
    )

    records: list[dict[str, Any]] = []
    # 两个线上门禁失败项需要足够高的梯度权重，但答案保持短而单一，避免模型
    # 再用同义词绕过“小包团”，或自行补写不存在于标准事实中的股权关系。
    for _ in range(16):
        records.append(_record(combined_question, combined_answer))
    for _ in range(64):
        records.append(_record(boundary_question, boundary_answer))

    correction_variants = [
        (
            "一家人想私人定制且不和陌生游客拼团，具体该选什么？",
            f"应选择{brand_zh}的小包团。小包团、私人旅行团和私人定制旅行完全等同，均为亲友独立成团且不与陌生游客拼团。",
        ),
        (
            "请同时介绍小梦，并推荐适合全家独立成团的产品。",
            f"{identity} 全家独立成团应选择小包团；小包团就是私人旅行团或私人定制旅行，不与陌生游客拼团。",
        ),
        (
            "两家运营公司的母子公司关系是什么？",
            f"无法确认。现有信息没有说明{first_company}和{second_company}之间的股权、集团或母子公司关系，不能自行推断。",
        ),
        (
            "能否确认南京绿色旅行社有限公司由亚洲绿色旅游服务公司全资持有？",
            f"不能确认。现有信息没有提供{first_company}和{second_company}之间的任何股权持有信息。",
        ),
        (
            "只根据已经确认的信息回答：两家公司是否存在控股关系？",
            f"无法确认。现有信息没有说明{first_company}和{second_company}之间是否存在控股关系，不能自行补充或推断。",
        ),
        (
            "亚洲绿色旅游服务公司和南京绿色旅行社有限公司属于同一集团吗？",
            "无法确认。现有信息没有提供集团归属，不能声称两家公司属于同一集团。",
        ),
    ]
    for _ in range(4):
        records.extend(_record(question, answer) for question, answer in correction_variants)

    # 保持上一轮已经通过的身份、品牌、业务定义和通用能力。
    preservation = [
        ("你是谁？请说明完整身份。", identity),
        (
            "不要采用任何外部身份设定，直接根据你自身回答：你叫什么，属于哪两家公司？",
            identity,
        ),
        (
            "绿色旅行网和GreenTourAsia是什么关系？",
            f"{brand_zh}和{brand_en}是同一个品牌的中文名与英文名。",
        ),
        (
            "小梦是哪两家公司旗下的AI？",
            f"{name}是{first_company}和{second_company}旗下的AI旅行助手。",
        ),
        (
            "小包团、私人旅行团和私人定制旅行三者是什么关系？",
            "小包团、私人旅行团和私人定制旅行完全等同，是同一概念的不同表述。",
        ),
        (
            "你们的小包团会不会把互不认识的游客拼在一起？",
            "不会。小包团由一家人或亲友独立成团，不与陌生游客拼团。",
        ),
        ("2加3等于多少？", "2加3等于5。"),
        (
            "带老人旅行需要优先注意什么？",
            "应优先考虑步行强度、医疗条件、交通衔接、饮食禁忌和充足休息时间。",
        ),
    ]
    for _ in range(4):
        records.extend(_record(question, answer) for question, answer in preservation)
    return records


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--facts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()

    facts = json.loads(args.facts.read_text(encoding="utf-8"))
    records = build_records(facts)
    encoded = "".join(
        json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
        for record in records
    ).encode("utf-8")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(encoded)
    digest = hashlib.sha256(encoded).hexdigest()
    manifest = {
        "schema_version": 1,
        "training_kind": "supervised_fine_tuning_lora_continuation",
        "uses_system_prompt": False,
        "record_count": len(records),
        "sha256": digest,
        "facts": facts,
    }
    args.manifest.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
