#!/usr/bin/env bash
set -euo pipefail

readonly common_config=/opt/gta-ai-image-embedding/config/vision.env
readonly node_config=/opt/gta-ai-image-embedding/config/node.env

for config_file in "$common_config" "$node_config"; do
  if [[ ! -r "$config_file" ]]; then
    echo "缺少可读配置文件：$config_file" >&2
    exit 1
  fi
done

# shellcheck disable=SC1090
source "$common_config"
# shellcheck disable=SC1090
source "$node_config"

required_variables=(
  VISION_IMAGE VISION_PORT VISION_DATA_DIR VISION_MODEL_ID VISION_MODEL_REVISION
  VISION_MEMORY_LIMIT VISION_INTEROP_THREADS VISION_MAX_IMAGE_BYTES
  VISION_MAX_IMAGE_PIXELS VISION_MAX_IMAGE_BATCH VISION_MAX_TEXT_BATCH
  SERVICE_HOST VISION_CPU_LIMIT VISION_CPU_THREADS
)
for variable_name in "${required_variables[@]}"; do
  if [[ -z "${!variable_name:-}" ]]; then
    echo "配置项不能为空：$variable_name" >&2
    exit 1
  fi
done

if ! ip -4 -o addr show | awk '{print $4}' | cut -d/ -f1 | grep -Fxq "$SERVICE_HOST"; then
  echo "SERVICE_HOST 不是本机地址：$SERVICE_HOST" >&2
  exit 1
fi

install -d -m 0750 "$VISION_DATA_DIR" "$VISION_DATA_DIR/home" "$VISION_DATA_DIR/huggingface"

exec /usr/bin/podman run --rm \
  --name gta-ai-image-embedding \
  --network host \
  --cpus "$VISION_CPU_LIMIT" \
  --memory "$VISION_MEMORY_LIMIT" \
  --memory-swap "$VISION_MEMORY_LIMIT" \
  --pids-limit 1024 \
  --read-only \
  --tmpfs /tmp:rw,noexec,nosuid,size=1g \
  --cap-drop all \
  --security-opt no-new-privileges \
  --env "HOME=$VISION_DATA_DIR/home" \
  --env "HF_HOME=$VISION_DATA_DIR/huggingface" \
  --env HF_HUB_DISABLE_XET=1 \
  --env "OMP_NUM_THREADS=$VISION_CPU_THREADS" \
  --env "MKL_NUM_THREADS=$VISION_CPU_THREADS" \
  --env "VISION_CPU_THREADS=$VISION_CPU_THREADS" \
  --env "VISION_INTEROP_THREADS=$VISION_INTEROP_THREADS" \
  --env "VISION_MODEL_CACHE=$VISION_DATA_DIR/huggingface" \
  --env "VISION_MODEL_ID=$VISION_MODEL_ID" \
  --env "VISION_MODEL_REVISION=$VISION_MODEL_REVISION" \
  --env "VISION_MAX_IMAGE_BYTES=$VISION_MAX_IMAGE_BYTES" \
  --env "VISION_MAX_IMAGE_PIXELS=$VISION_MAX_IMAGE_PIXELS" \
  --env "VISION_MAX_IMAGE_BATCH=$VISION_MAX_IMAGE_BATCH" \
  --env "VISION_MAX_TEXT_BATCH=$VISION_MAX_TEXT_BATCH" \
  --volume "$VISION_DATA_DIR:$VISION_DATA_DIR:rw" \
  "$VISION_IMAGE" \
  --host "$SERVICE_HOST" \
  --port "$VISION_PORT" \
  --workers 1 \
  --limit-concurrency 4 \
  --timeout-keep-alive 30 \
  --no-access-log
