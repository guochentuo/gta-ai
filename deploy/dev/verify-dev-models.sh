#!/usr/bin/env bash
set -euo pipefail

declare -A endpoints=(
  [QwenEmbedding]="http://192.168.80.2:18080/health"
  [OCR]="http://192.168.80.2:18081/health"
  [SigLIP2]="http://192.168.80.2:18082/health"
  [ASR]="http://192.168.80.2:18084/health"
  [AST]="http://192.168.80.2:18085/health"
)

for model in QwenEmbedding OCR SigLIP2 ASR AST; do
  endpoint="${endpoints[$model]}"
  code="$(curl --silent --show-error --output /dev/null --write-out '%{http_code}' \
    --connect-timeout 3 --max-time 10 "$endpoint")"
  if [[ "$code" != "200" ]]; then
    printf '[ERROR] %s health check failed: endpoint=%s http=%s\n' "$model" "$endpoint" "$code" >&2
    exit 1
  fi
  printf '[OK] %s %s\n' "$model" "$endpoint"
done
