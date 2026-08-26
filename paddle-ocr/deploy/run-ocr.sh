#!/usr/bin/env bash
set -euo pipefail

readonly default_config_files=/opt/gta-ai/paddle-ocr/config/ocr.env,/opt/gta-ai/paddle-ocr/config/node.env
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
  OCR_IMAGE OCR_PORT OCR_DATA_DIR OCR_DETECTION_MODEL OCR_RECOGNITION_MODEL
  OCR_MEMORY_LIMIT OCR_MAX_IMAGE_BYTES OCR_MAX_IMAGE_PIXELS OCR_MIN_SCORE OCR_MAX_BATCH_SIZE
  SERVICE_HOST OCR_CPU_LIMIT OCR_CPU_THREADS
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

install -d -m 0750 "$OCR_DATA_DIR" "$OCR_DATA_DIR/home" "$OCR_DATA_DIR/paddlex"

# 推理锁仍保证单模型串行；允许多个批次在服务端排队，避免并行视频被直接 503。
exec /usr/bin/podman run --rm \
  --name gta-ai-ocr \
  --network host \
  --cpus "$OCR_CPU_LIMIT" \
  --memory "$OCR_MEMORY_LIMIT" \
  --memory-swap "$OCR_MEMORY_LIMIT" \
  --pids-limit 1024 \
  --shm-size 1g \
  --read-only \
  --tmpfs /tmp:rw,noexec,nosuid,size=1g \
  --cap-drop all \
  --security-opt no-new-privileges \
  --env "HOME=$OCR_DATA_DIR/home" \
  --env "PADDLE_PDX_CACHE_HOME=$OCR_DATA_DIR/paddlex" \
  --env PADDLE_PDX_MODEL_SOURCE=BOS \
  --env "OMP_NUM_THREADS=$OCR_CPU_THREADS" \
  --env "MKL_NUM_THREADS=$OCR_CPU_THREADS" \
  --env "OCR_CPU_THREADS=$OCR_CPU_THREADS" \
  --env "OCR_DETECTION_MODEL=$OCR_DETECTION_MODEL" \
  --env "OCR_RECOGNITION_MODEL=$OCR_RECOGNITION_MODEL" \
  --env "OCR_MAX_IMAGE_BYTES=$OCR_MAX_IMAGE_BYTES" \
  --env "OCR_MAX_IMAGE_PIXELS=$OCR_MAX_IMAGE_PIXELS" \
  --env "OCR_MIN_SCORE=$OCR_MIN_SCORE" \
  --env "OCR_MAX_BATCH_SIZE=$OCR_MAX_BATCH_SIZE" \
  --volume "$OCR_DATA_DIR:$OCR_DATA_DIR:rw" \
  "$OCR_IMAGE" \
  --host "$SERVICE_HOST" \
  --port "$OCR_PORT" \
  --workers 1 \
  --limit-concurrency 8 \
  --timeout-keep-alive 30 \
  --no-access-log
