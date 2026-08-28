#!/usr/bin/env bash
set -euo pipefail

app_root=/opt/gta-ai/qwen-llm
config_file=$app_root/config/training/finetune.env
. "$config_file"
data_root=/opt/gta-ai/27b/training
dataset=$data_root/datasets/identity-balanced.jsonl
manifest=$data_root/datasets/identity-balanced.manifest.json
preservation=$data_root/datasets/preservation.jsonl
pending=$data_root/queue/pending.json
facts=$app_root/training/identity/canonical.json

mkdir -p "$(dirname -- "$dataset")" "$(dirname -- "$pending")"
test -s "$preservation"
"$TRAIN_PYTHON" "$app_root/training/identity/build_balanced_dataset.py" \
    --facts "$facts" \
    --preservation "$preservation" \
    --output "$dataset" \
    --manifest "$manifest"

"$TRAIN_PYTHON" - "$manifest" "$dataset" "$pending" <<'PY'
import json
import os
import sys
from pathlib import Path

manifest_path, dataset_path, pending_path = map(Path, sys.argv[1:])
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
revision = f"identity-{manifest['sha256'][:16]}"
pending = {
    "schema_version": 1,
    "revision": revision,
    "training_kind": manifest["training_kind"],
    "uses_system_prompt": False,
    "dataset_path": str(dataset_path),
    "dataset_sha256": manifest["sha256"],
    "record_count": manifest["record_count"],
}
target = Path(pending_path)
temporary = target.with_suffix(".tmp")
temporary.write_text(json.dumps(pending, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
os.replace(temporary, target)
print(json.dumps(pending, ensure_ascii=False))
PY
