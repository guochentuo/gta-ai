#!/usr/bin/env bash
set -euo pipefail

readonly default_config_files=/opt/gta-ai/embedding/config/embedding.env,/opt/gta-ai/embedding/config/node.env
IFS=',' read -r -a config_files <<<"${GTA_AI_CONFIG_FILES:-$default_config_files}"

for config_file in "${config_files[@]}"; do
  if [[ ! -r "$config_file" ]]; then
    echo "缺少可读配置文件：$config_file" >&2
    exit 1
  fi
done

# shellcheck disable=SC1090
for config_file in "${config_files[@]}"; do
  # shellcheck disable=SC1090
  source "$config_file"
done

required_variables=(
  TEI_IMAGE
  TEI_MODEL_ID
  TEI_MODEL_REVISION
  TEI_HOST
  TEI_PORT
  TEI_DTYPE
  TEI_DATA_DIR
  TEI_CPU_LIMIT
  TEI_MEMORY_LIMIT
  TEI_MAX_CONCURRENT_REQUESTS
  TEI_MAX_CLIENT_BATCH_SIZE
  TEI_MAX_BATCH_TOKENS
  TEI_THREADS
)

for variable_name in "${required_variables[@]}"; do
  if [[ -z "${!variable_name:-}" ]]; then
    echo "配置项不能为空：$variable_name" >&2
    exit 1
  fi
done

if ! ip -4 -o addr show | awk '{print $4}' | cut -d/ -f1 | grep -Fxq "$TEI_HOST"; then
  echo "TEI_HOST 不是本机地址：$TEI_HOST" >&2
  exit 1
fi

mkdir -p "$TEI_DATA_DIR"

exec /usr/bin/podman run --rm \
  --name gta-ai-embedding \
  --network host \
  --cpus "$TEI_CPU_LIMIT" \
  --memory "$TEI_MEMORY_LIMIT" \
  --memory-swap "$TEI_MEMORY_LIMIT" \
  --pids-limit 512 \
  --cap-drop all \
  --security-opt no-new-privileges \
  --env "OMP_NUM_THREADS=$TEI_THREADS" \
  --env "MKL_NUM_THREADS=$TEI_THREADS" \
  --env "RAYON_NUM_THREADS=$TEI_THREADS" \
  --env HF_HUB_OFFLINE=1 \
  --volume "$TEI_DATA_DIR:/data:rw" \
  "$TEI_IMAGE" \
  --model-id "$TEI_MODEL_ID" \
  --revision "$TEI_MODEL_REVISION" \
  --hostname "$TEI_HOST" \
  --port "$TEI_PORT" \
  --dtype "$TEI_DTYPE" \
  --max-concurrent-requests "$TEI_MAX_CONCURRENT_REQUESTS" \
  --max-client-batch-size "$TEI_MAX_CLIENT_BATCH_SIZE" \
  --max-batch-tokens "$TEI_MAX_BATCH_TOKENS" \
  --auto-truncate
