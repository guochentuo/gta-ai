#!/usr/bin/env bash
set -euo pipefail

config_file=/opt/gta-ai/qwen-llm/config/training/finetune.env
. "$config_file"

model_complete() {
    python3 - "$TRAIN_MODEL_PATH" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
index = root / "model.safetensors.index.json"
if not (root / "config.json").is_file() or not index.is_file():
    raise SystemExit(1)
weight_map = json.loads(index.read_text(encoding="utf-8")).get("weight_map", {})
shards = {root / str(name) for name in weight_map.values()}
if not shards or any(not shard.is_file() or shard.stat().st_size == 0 for shard in shards):
    raise SystemExit(1)
PY
}

if model_complete; then
    echo "Training model is already complete: $TRAIN_MODEL_PATH"
    exit 0
fi

mkdir -p "$(dirname -- "$TRAIN_MODEL_PATH")" /opt/gta-ai/data/cache/huggingface
/usr/bin/podman run --rm \
    --entrypoint python3 \
    --security-opt=label=disable \
    --volume "$(dirname -- "$TRAIN_MODEL_PATH"):/models:rw" \
    --volume /opt/gta-ai/data/cache/huggingface:/root/.cache/huggingface:rw \
    "$TRAIN_IMAGE" \
    -c 'from huggingface_hub import snapshot_download; snapshot_download(repo_id="Qwen/Qwen3.6-27B", local_dir="/models/Qwen3.6-27B")'

model_complete
