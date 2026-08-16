#!/usr/bin/env bash
set -euo pipefail

readonly endpoint="${1:-http://192.168.80.130}"
readonly image_path="${2:?用法：verify-image-services.sh ENDPOINT IMAGE_PATH}"

curl --fail --silent --show-error --max-time 10 "$endpoint:18081/health" | jq -e '.status == "ok"' >/dev/null
curl --fail --silent --show-error --max-time 10 "$endpoint:18082/health" | jq -e '.status == "ok"' >/dev/null

ocr_response="$(curl --fail --silent --show-error --max-time 180 \
  "$endpoint:18081/v1/ocr" -F "file=@$image_path")"
jq -e '.source_sha256 | length == 64' <<<"$ocr_response" >/dev/null
ocr_batch_response="$(curl --fail --silent --show-error --max-time 180 \
  "$endpoint:18081/v1/ocr-batch" \
  -F "files=@$image_path" -F "files=@$image_path")"
jq -e '.count == 2 and (.items | length == 2) and
       ([.items[].source_sha256 | length == 64] | all)' \
  <<<"$ocr_batch_response" >/dev/null

image_response="$(curl --fail --silent --show-error --max-time 180 \
  "$endpoint:18082/v1/image-embeddings" -F "files=@$image_path")"
text_response="$(curl --fail --silent --show-error --max-time 180 \
  "$endpoint:18082/v1/text-embeddings" \
  -H 'Content-Type: application/json' \
  -d '{"input":["江南水乡古镇","城市夜景"]}')"

image_dimensions="$(jq '.data[0].embedding | length' <<<"$image_response")"
text_dimensions="$(jq '.data[0].embedding | length' <<<"$text_response")"
declared_dimensions="$(jq '.dimensions' <<<"$image_response")"

if [[ "$image_dimensions" -le 0 || "$image_dimensions" != "$text_dimensions" || "$image_dimensions" != "$declared_dimensions" ]]; then
  echo "图片与文字向量维度不一致：image=$image_dimensions text=$text_dimensions declared=$declared_dimensions" >&2
  exit 1
fi

echo "图片服务验证通过：endpoint=$endpoint OCR_lines=$(jq '.lines | length' <<<"$ocr_response") OCR_batch=$(jq '.count' <<<"$ocr_batch_response") SigLIP_dims=$image_dimensions"
