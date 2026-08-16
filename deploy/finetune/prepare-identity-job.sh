#!/usr/bin/env bash
set -euo pipefail

app_root=/opt/gta-ai
dataset=$app_root/training/datasets/identity.jsonl
manifest=$app_root/training/datasets/identity.manifest.json
pending=$app_root/training/queue/pending.json
facts=$app_root/training/identity/canonical.json

mkdir -p "$(dirname -- "$dataset")" "$(dirname -- "$pending")"
"$app_root/app/.venv/bin/python" "$app_root/training/identity/build_dataset.py" \
    --facts "$facts" \
    --output "$dataset" \
    --manifest "$manifest"

"$app_root/app/.venv/bin/python" - "$manifest" "$dataset" "$pending" <<'PY'
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
