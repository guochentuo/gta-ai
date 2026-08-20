#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PACKAGE_ROOT="${PACKAGE_ROOT:-$ROOT/release}"
STAMP="$(date -u +%Y%m%d_%H%M%S)"
REV="$(git -C "$ROOT" rev-parse HEAD)"
ARCHIVE="gta-ai-${STAMP}-${REV:0:12}.tar.gz"
mkdir -p "$PACKAGE_ROOT"
STAGING="$(mktemp -d "$PACKAGE_ROOT/.gta-ai.XXXXXX")"
trap 'rm -rf -- "$STAGING"' EXIT
PKG="$STAGING/gta-ai"
install -d -m 0755 "$PKG"/{bin,systemd,logs}
install -d -m 0755 "$PKG"/{27b/{bin,config,systemd},router/{bin,config,systemd,src/gta_ai}}
for model in embedding ocr siglip ast asr; do
  install -d -m 0755 "$PKG/$model/config"
done
install -m 0755 "$ROOT/deploy/common/json_log_runner.py" "$PKG/bin/json-log-runner.py"
install -m 0755 "$ROOT/deploy/embedding-cluster/run-embedding.sh" "$PKG/bin/run-embedding"
install -m 0755 "$ROOT/deploy/image-inference-cluster/run-ocr.sh" "$PKG/bin/run-ocr"
install -m 0755 "$ROOT/deploy/image-inference-cluster/run-vision.sh" "$PKG/bin/run-vision"
install -m 0755 "$ROOT/deploy/audio-inference/run-audio-events" "$PKG/bin/run-audio-events"
install -m 0755 "$ROOT/deploy/dev/verify-dev-models.sh" "$PKG/bin/verify-dev-models"
install -m 0644 "$ROOT/deploy/asr-inference/app.py" "$PKG/asr/app.py"
install -m 0600 "$ROOT/deploy/dev/embedding.dev.env" "$PKG/embedding/config/"
install -m 0600 "$ROOT/deploy/dev/ocr.dev.env" "$PKG/ocr/config/"
install -m 0600 "$ROOT/deploy/dev/vision.dev.env" "$PKG/siglip/config/"
install -m 0600 "$ROOT/deploy/dev/audio-events.dev.env" "$PKG/ast/config/"
install -m 0600 "$ROOT/deploy/dev/asr.dev.env" "$PKG/asr/config/"
install -m 0644 "$ROOT"/deploy/dev/*.service "$PKG/systemd/"
install -m 0644 "$ROOT/README.md" "$PKG/README.md"
install -m 0755 "$ROOT/deploy/dev/install-dev-package.sh" "$PKG/"
install -m 0755 "$ROOT/deploy/run-vllm.sh" "$PKG/27b/bin/run-vllm"
install -m 0600 "$ROOT/deploy/vllm.env" "$PKG/27b/config/vllm.env"
install -m 0644 "$ROOT/deploy/gta-ai-vllm.service" "$PKG/27b/systemd/"
install -m 0755 "$ROOT/deploy/common/json_log_runner.py" "$PKG/router/bin/json-log-runner.py"
install -m 0755 "$ROOT/deploy/finetune/run-router.sh" "$PKG/router/bin/run-router"
install -m 0600 "$ROOT/deploy/finetune/router.env" "$PKG/router/config/router.env"
install -m 0644 "$ROOT/deploy/finetune/gta-ai-router.service" "$PKG/router/systemd/"
install -m 0644 "$ROOT/src/gta_ai/__init__.py" "$ROOT/src/gta_ai/inference_router.py" \
  "$PKG/router/src/gta_ai/"
install -m 0755 "$ROOT/deploy/finetune/install-27b-package.sh" "$PKG/"
cat > "$PKG/RELEASE-MANIFEST.json" <<EOF
{"schemaVersion":1,"project":"gta-ai","gitRevision":"$REV","gitState":"$([[ -n "$(git -C "$ROOT" status --porcelain)" ]] && echo dirty || echo clean)","buildTimeUtc":"$STAMP","installRoot":"/opt/gta-ai","deploymentSource":"verified-project-package-only","weightsIncluded":false}
EOF
(cd "$PKG" && find . -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS)
tar -C "$STAGING" --owner=0 --group=0 --numeric-owner -czf "$PACKAGE_ROOT/$ARCHIVE" gta-ai
(cd "$PACKAGE_ROOT" && sha256sum "$ARCHIVE" > "$ARCHIVE.sha256")
printf '%s\n' "$PACKAGE_ROOT/$ARCHIVE"
