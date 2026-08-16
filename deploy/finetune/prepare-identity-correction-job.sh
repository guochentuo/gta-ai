#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 1 ]; then
    echo "usage: $0 PARENT_REVISION" >&2
    exit 2
fi

app_root=/opt/gta-ai
parent_revision=$1
parent_completion=$app_root/training/runtime/$parent_revision/training-complete.json
dataset=$app_root/training/datasets/identity-correction.jsonl
manifest=$app_root/training/datasets/identity-correction.manifest.json
pending=$app_root/training/queue/pending.json
facts=$app_root/training/identity/canonical.json

test -f "$parent_completion"
mkdir -p "$(dirname -- "$dataset")" "$(dirname -- "$pending")"
"$app_root/app/.venv/bin/python" "$app_root/training/identity/build_correction_dataset.py" \
    --facts "$facts" \
    --output "$dataset" \
    --manifest "$manifest"

"$app_root/app/.venv/bin/python" - \
    "$manifest" "$dataset" "$pending" "$parent_completion" "$parent_revision" <<'PY'
import hashlib
import json
import os
import sys
from pathlib import Path

manifest_path, dataset_path, pending_path, parent_completion_path = map(Path, sys.argv[1:5])
parent_revision = sys.argv[5]
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
parent_completion = json.loads(parent_completion_path.read_text(encoding="utf-8"))
parent_adapter = Path(parent_completion["adapter_path"]).resolve()
adapter_root = Path("/opt/gta-ai/training/adapters").resolve()
if not parent_adapter.is_relative_to(adapter_root):
    raise SystemExit("parent adapter is outside the managed adapter root")
if not (parent_adapter / "adapter_config.json").is_file():
    raise SystemExit("parent adapter is incomplete")
lineage = hashlib.sha256(
    f"{manifest['sha256']}\0{parent_revision}\0{parent_adapter}".encode()
).hexdigest()
revision = f"identity-correction-{lineage[:16]}"
pending = {
    "schema_version": 1,
    "revision": revision,
    "training_kind": manifest["training_kind"],
    "uses_system_prompt": False,
    "dataset_path": str(dataset_path),
    "dataset_sha256": manifest["sha256"],
    "record_count": manifest["record_count"],
    "parent_revision": parent_revision,
    "parent_adapter_path": str(parent_adapter),
}
target = pending_path
temporary = target.with_suffix(".tmp")
temporary.write_text(json.dumps(pending, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
os.replace(temporary, target)
print(json.dumps(pending, ensure_ascii=False))
PY
