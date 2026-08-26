from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from evaluate_identity import score_text


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    definitions = json.loads(args.cases.read_text(encoding="utf-8"))
    source = json.loads(args.source.read_text(encoding="utf-8"))
    if source.get("uses_system_prompt") is not False:
        raise SystemExit("source report was not generated without a system prompt")

    cases = definitions["hard_cases"] + definitions.get("smoke_cases", [])
    source_by_id = {result["id"]: result for result in source["results"]}
    if set(source_by_id) != {case["id"] for case in cases}:
        raise SystemExit("source report case set does not match current definitions")

    results = []
    for case in cases:
        previous = source_by_id[case["id"]]
        text = str(previous["response"])
        results.append(
            {
                "id": case["id"],
                "elapsed_ms": previous.get("elapsed_ms", 0),
                "response": text,
                **score_text(text, case),
            }
        )

    hard_ids = {case["id"] for case in definitions["hard_cases"]}
    hard_passed = all(result["passed"] for result in results if result["id"] in hard_ids)
    smoke_passed = all(result["passed"] for result in results if result["id"] not in hard_ids)
    report = {
        "schema_version": 1,
        "uses_system_prompt": False,
        "model": source["model"],
        "hard_passed": hard_passed,
        "smoke_passed": smoke_passed,
        "passed": hard_passed and smoke_passed,
        "rescored_from_existing_responses": True,
        "results": results,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.report.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, args.report)
    print(json.dumps(report, ensure_ascii=False))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
