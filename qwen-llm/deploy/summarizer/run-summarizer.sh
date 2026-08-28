#!/usr/bin/env bash
set -eu

config_file=/opt/gta-ai/qwen-llm/config/summarizer.env
. "$config_file"

if [ ! -f "$SUMMARIZER_MODEL_PATH" ]; then
    echo "摘要模型不存在: $SUMMARIZER_MODEL_PATH" >&2
    exit 1
fi

exec /usr/bin/podman run --rm \
    --name gta-ai-qwen-summarizer \
    --security-opt=label=disable \
    --log-driver=journald \
    --publish "$SUMMARIZER_HOST:$SUMMARIZER_PORT:8080" \
    --volume "$SUMMARIZER_MODEL_PATH:/models/summarizer.gguf:ro" \
    "$SUMMARIZER_IMAGE" \
    --model /models/summarizer.gguf \
    --alias "$SUMMARIZER_MODEL_NAME" \
    --host 0.0.0.0 \
    --port 8080 \
    --threads "$SUMMARIZER_THREADS" \
    --threads-batch "$SUMMARIZER_BATCH_THREADS" \
    --ctx-size "$SUMMARIZER_CONTEXT_SIZE" \
    --parallel "$SUMMARIZER_PARALLEL" \
    --device none \
    --metrics \
    --no-webui
