#!/usr/bin/env sh
set -eu

config_file=/opt/gta-ai/config/vllm.env

if [ ! -r "$config_file" ]; then
    echo "Missing readable configuration: $config_file" >&2
    exit 1
fi

# shellcheck disable=SC1090
. "$config_file"

if [ ! -f "$VLLM_MODEL_PATH/config.json" ]; then
    echo "Model snapshot is incomplete: $VLLM_MODEL_PATH/config.json is missing" >&2
    exit 1
fi

exec /usr/bin/podman run --rm \
    --name gta-ai-vllm \
    --device nvidia.com/gpu=all \
    --security-opt=label=disable \
    --ipc=host \
    --publish "$VLLM_HOST:$VLLM_PORT:8000" \
    --volume "$VLLM_MODEL_PATH:/models/Qwen3.6-27B-FP8:ro" \
    --volume /opt/gta-ai/data/cache/huggingface:/root/.cache/huggingface:rw \
    --volume /opt/gta-ai/data/cache/vllm:/root/.cache/vllm:rw \
    "$VLLM_IMAGE" \
    /models/Qwen3.6-27B-FP8 \
    --served-model-name "$VLLM_SERVED_MODEL_NAME" \
    --host 0.0.0.0 \
    --port 8000 \
    --max-model-len "$VLLM_MAX_MODEL_LEN" \
    --max-num-seqs "$VLLM_MAX_NUM_SEQS" \
    --kv-cache-dtype "$VLLM_KV_CACHE_DTYPE" \
    --gpu-memory-utilization "$VLLM_GPU_MEMORY_UTILIZATION" \
    --enforce-eager \
    --reasoning-parser qwen3 \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_coder
