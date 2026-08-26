#!/usr/bin/env bash
set -euo pipefail

config_file=/opt/gta-ai/qwen-llm/config/training/finetune.env
. "$config_file"

training_container=gta-ai-identity-trainer
validator_container=gta-ai-identity-validator
cleanup() {
    /usr/bin/podman rm --force --time 0 "$training_container" >/dev/null 2>&1 || true
    /usr/bin/podman rm --force --time 0 "$validator_container" >/dev/null 2>&1 || true
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

mapfile -t job < <(
    /opt/gta-ai/qwen-llm/.venv/bin/python - "$TRAIN_PENDING_MANIFEST" "$TRAIN_DATASET_PATH" <<'PY'
import hashlib
import json
import re
import sys
from pathlib import Path

manifest_path, configured_dataset = map(Path, sys.argv[1:])
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
revision = str(manifest["revision"])
dataset = Path(str(manifest["dataset_path"])).resolve()
expected_hash = str(manifest["dataset_sha256"])
if not re.fullmatch(r"[A-Za-z0-9._-]{1,100}", revision):
    raise SystemExit("invalid training revision")
dataset_root = configured_dataset.resolve().parent
if not dataset.is_relative_to(dataset_root):
    raise SystemExit("pending job points outside the managed dataset root")
actual_hash = hashlib.sha256(dataset.read_bytes()).hexdigest()
if actual_hash != expected_hash:
    raise SystemExit("training dataset hash mismatch")
if manifest.get("uses_system_prompt") is not False:
    raise SystemExit("identity training must not depend on a system prompt")
print(revision)
print(expected_hash)
print(dataset)
parent_adapter = str(manifest.get("parent_adapter_path", ""))
if parent_adapter:
    parent = Path(parent_adapter).resolve()
    adapter_root = Path("/opt/gta-ai/qwen-llm/data/training/adapters").resolve()
    if not parent.is_relative_to(adapter_root):
        raise SystemExit("parent adapter is outside the managed adapter root")
    if not (parent / "adapter_config.json").is_file():
        raise SystemExit("parent adapter is incomplete")
    parent_adapter = str(parent)
print(parent_adapter)
PY
)
revision=${job[0]}
dataset_hash=${job[1]}
dataset_path=${job[2]}
parent_adapter=${job[3]:-}
output_dir=$TRAIN_OUTPUT_ROOT/$revision
runtime_dir=$TRAIN_RUNTIME_ROOT/$revision
completion_file=$runtime_dir/training-complete.json
report_file=$runtime_dir/evaluation.json
candidate_file=$TRAIN_RUNTIME_ROOT/candidate.json
log_file=/opt/gta-ai/logs/gta.ai.training.log

mkdir -p "$output_dir" "$runtime_dir" "$(dirname -- "$log_file")"
exec > >(tee -a "$log_file") 2>&1
echo "[$(date --iso-8601=seconds)] identity fine-tuning revision=$revision"

adapter_path=
if [ -f "$completion_file" ]; then
    adapter_path=$(
        /opt/gta-ai/qwen-llm/.venv/bin/python - "$completion_file" <<'PY'
import json
import sys
print(json.load(open(sys.argv[1], encoding="utf-8"))["adapter_path"])
PY
    )
fi

if [ -z "$adapter_path" ] || [ ! -f "$adapter_path/adapter_config.json" ]; then
    resume_args=()
    latest_checkpoint=$(find "$output_dir" -type f -name adapter_config.json -printf '%T@ %h\n' 2>/dev/null | sort -nr | head -n 1 | cut -d' ' -f2- || true)
    if [ -n "$latest_checkpoint" ]; then
        resume_args=(--resume_from_checkpoint "/output/${latest_checkpoint#"$output_dir"/}")
    fi

    cleanup
    parent_mount=()
    adapter_args=()
    learning_rate=3e-5
    if [ -n "$parent_adapter" ]; then
        parent_mount=(--volume "$parent_adapter:/models/parent-adapter:ro")
        adapter_args=(--adapters /models/parent-adapter)
        learning_rate=1e-5
    fi
    /usr/bin/podman run --rm \
        --name "$training_container" \
        --device nvidia.com/gpu=all \
        --security-opt=label=disable \
        --ipc=host \
        --env PYTORCH_ALLOC_CONF=expandable_segments:True \
        --env HF_DEACTIVATE_ASYNC_LOAD=1 \
        --env IMAGE_MAX_TOKEN_NUM=256 \
        --env VIDEO_MAX_TOKEN_NUM=16 \
        --volume "$TRAIN_MODEL_PATH:/models/Qwen3.6-27B:ro" \
        --volume "$dataset_path:/training/identity.jsonl:ro" \
        --volume "$output_dir:/output:rw" \
        "${parent_mount[@]}" \
        --volume /opt/gta-ai/data/cache/huggingface:/root/.cache/huggingface:rw \
        --entrypoint swift \
        "$TRAIN_IMAGE" \
        sft \
        --model /models/Qwen3.6-27B \
        "${adapter_args[@]}" \
        --dataset /training/identity.jsonl \
        --tuner_type lora \
        --torch_dtype bfloat16 \
        --quant_method bnb \
        --quant_bits 4 \
        --bnb_4bit_compute_dtype bfloat16 \
        --bnb_4bit_quant_type nf4 \
        --bnb_4bit_use_double_quant true \
        --freeze_vit true \
        --freeze_aligner true \
        --target_modules all-linear \
        --lora_rank 8 \
        --lora_alpha 16 \
        --lora_dropout 0.05 \
        --add_non_thinking_prefix true \
        --loss_scale ignore_empty_think \
        --num_train_epochs 2 \
        --per_device_train_batch_size 1 \
        --per_device_eval_batch_size 1 \
        --gradient_accumulation_steps 8 \
        --learning_rate "$learning_rate" \
        --warmup_ratio 0.05 \
        --max_length 512 \
        --gradient_checkpointing true \
        --split_dataset_ratio 0.05 \
        --eval_steps 25 \
        --save_steps 25 \
        --save_total_limit 2 \
        --logging_steps 5 \
        --dataset_num_proc 4 \
        --dataloader_num_workers 2 \
        --report_to none \
        --output_dir /output \
        "${resume_args[@]}"

    adapter_path=$(find "$output_dir" -type f -name adapter_config.json -printf '%T@ %h\n' | sort -nr | head -n 1 | cut -d' ' -f2-)
    test -n "$adapter_path"
    /opt/gta-ai/qwen-llm/.venv/bin/python - "$completion_file" "$adapter_path" "$revision" <<'PY'
import json
import os
import sys
from pathlib import Path

target, adapter, revision = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
temporary = target.with_suffix(".tmp")
temporary.write_text(
    json.dumps({"revision": revision, "adapter_path": adapter}, ensure_ascii=False) + "\n",
    encoding="utf-8",
)
os.replace(temporary, target)
PY
fi

test -f "$adapter_path/adapter_config.json"
cleanup
/usr/bin/podman run --detach --rm \
    --name "$validator_container" \
    --device nvidia.com/gpu=all \
    --security-opt=label=disable \
    --ipc=host \
    --publish "127.0.0.1:$TRAIN_VALIDATION_PORT:8000" \
    --volume /opt/gta-ai/models/Qwen3.6-27B-FP8:/models/Qwen3.6-27B-FP8:ro \
    --volume "$adapter_path:/models/identity-adapter:ro" \
    --volume /opt/gta-ai/data/cache/huggingface:/root/.cache/huggingface:rw \
    --volume /opt/gta-ai/data/cache/vllm:/root/.cache/vllm:rw \
    docker.io/vllm/vllm-openai@sha256:7a0f0fdd2771464b6976625c2b2d5dd46f566aa00fbc53eceab86ef50883da90 \
    /models/Qwen3.6-27B-FP8 \
    --served-model-name "$TRAIN_BASE_MODEL_NAME" \
    --host 0.0.0.0 \
    --port 8000 \
    --max-model-len 32768 \
    --max-num-seqs 1 \
    --kv-cache-dtype fp8 \
    --gpu-memory-utilization 0.82 \
    --enforce-eager \
    --reasoning-parser qwen3 \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_coder \
    --enable-lora \
    --max-lora-rank "$TRAIN_MAX_LORA_RANK" \
    --lora-modules "$TRAIN_MODEL_NAME=/models/identity-adapter"

ready=false
for _ in $(seq 1 1200); do
    if curl --fail --silent "http://127.0.0.1:$TRAIN_VALIDATION_PORT/v1/models" >/dev/null; then
        ready=true
        break
    fi
    sleep 0.25
done
if [ "$ready" != true ]; then
    echo "validation inference did not become ready" >&2
    exit 1
fi

/opt/gta-ai/qwen-llm/.venv/bin/python "$TRAIN_EVAL_SCRIPT" \
    --cases "$TRAIN_EVAL_CASES" \
    --endpoint "http://127.0.0.1:$TRAIN_VALIDATION_PORT/v1/chat/completions" \
    --model "$TRAIN_MODEL_NAME" \
    --report "$report_file"

/opt/gta-ai/qwen-llm/.venv/bin/python - \
    "$candidate_file" "$revision" "$adapter_path" "$dataset_hash" "$report_file" <<'PY'
import json
import os
import sys
import time
from pathlib import Path

target, revision, adapter, dataset_hash, report = sys.argv[1:]
value = {
    "schema_version": 1,
    "revision": revision,
    "adapter_path": adapter,
    "dataset_sha256": dataset_hash,
    "evaluation_report": report,
    "uses_system_prompt": False,
    "validated_at": time.time(),
}
path = Path(target)
temporary = path.with_suffix(".tmp")
temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
os.replace(temporary, path)
PY
echo "[$(date --iso-8601=seconds)] identity adapter validated revision=$revision"
