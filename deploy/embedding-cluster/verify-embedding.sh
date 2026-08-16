#!/usr/bin/env bash
set -euo pipefail

readonly endpoint="${1:-http://192.168.80.130:18080}"
readonly model=Qwen/Qwen3-Embedding-0.6B

curl --fail --silent --show-error --max-time 10 "$endpoint/health" >/dev/null

response="$({
  curl --fail --silent --show-error --max-time 120 \
    "$endpoint/v1/embeddings" \
    -H 'Content-Type: application/json' \
    -d "{\"model\":\"$model\",\"input\":[\"杭州西湖旅游攻略\",\"三亚亲子度假行程\"],\"encoding_format\":\"float\"}"
})"

vector_count="$(jq '.data | length' <<<"$response")"
first_dimensions="$(jq '.data[0].embedding | length' <<<"$response")"
second_dimensions="$(jq '.data[1].embedding | length' <<<"$response")"

if [[ "$vector_count" != 2 || "$first_dimensions" != 1024 || "$second_dimensions" != 1024 ]]; then
  echo "Embedding 响应不符合预期：count=$vector_count dims=$first_dimensions,$second_dimensions" >&2
  exit 1
fi

echo "Embedding 验证通过：endpoint=$endpoint model=$model count=2 dims=1024"
