#!/usr/bin/env bash
set -euo pipefail

readonly default_config_files=/opt/gta-ai/qwen-embedding/config/embedding.env,/opt/gta-ai/qwen-embedding/config/node.env
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

# 日志心跳：TEI 是第三方二进制，无法在进程内发心跳；由本脚本的后台循环向同一 stdout
# （systemd 追加到 logs/service.log）每 300 秒写一条 JSON event=log_heartbeat，并附带本机 /health 结果。
# 后台循环随单元 cgroup 一起停止。
json_escape() { local value=${1//\\/\\\\}; printf '%s' "${value//\"/\\\"}"; }
heartbeat_interval=${LOG_HEARTBEAT_INTERVAL_SECONDS:-300}
log_env=$(json_escape "${GTA_ENV:-unknown}")
log_node=$(json_escape "${GTA_NODE:-$TEI_HOST}")
log_sha=$(json_escape "${GTA_DEPLOY_SHA:-unknown}")
log_host=$(json_escape "$(hostname)")
(
  started=$SECONDS
  while true; do
    health=unknown
    if command -v curl >/dev/null 2>&1; then
      if curl -fsS -o /dev/null --max-time 5 "http://$TEI_HOST:$TEI_PORT/health" 2>/dev/null; then
        health=ok
      else
        health=failed
      fi
    fi
    printf '{"@timestamp":"%s","env":"%s","service":"gta-ai-embedding","project":"gta-ai","node":"%s","host":"%s","deploy_sha":"%s","level":"INFO","logger":"run-embedding","event":"log_heartbeat","message":"log_heartbeat","fields":{"uptime_s":%d,"tei_health":"%s"}}\n' \
      "$(date -u +%Y-%m-%dT%H:%M:%S.%3NZ)" "$log_env" "$log_node" "$log_host" "$log_sha" \
      "$((SECONDS - started))" "$health"
    sleep "$heartbeat_interval"
  done
) &

# TEI_JSON_OUTPUT=true（默认）时 TEI 以 JSON 行输出日志（含每个请求的 "Success" 与耗时 span），
# 便于采集端按字段解析；设为 false 回退纯文本。
tei_log_args=()
if [[ "${TEI_JSON_OUTPUT:-true}" == "true" ]]; then
  tei_log_args+=(--json-output)
fi

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
  --auto-truncate \
  "${tei_log_args[@]}"
