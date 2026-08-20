#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
[[ -f "$ROOT/RELEASE-MANIFEST.json" && -d "$ROOT/27b" && -d "$ROOT/router" ]] || {
  echo "请从解压后的gta-ai发布包执行" >&2
  exit 2
}
[[ -f /opt/gta-ai/27b/models/Qwen3.6-27B-FP8/config.json ]] || {
  echo "缺少既有27B权重；安装器不会下载模型" >&2
  exit 2
}
[[ -x /opt/gta-ai/router/app/.venv/bin/python ]] || {
  echo "缺少既有router虚拟环境；安装器不会联网安装依赖" >&2
  exit 2
}

(cd "$ROOT" && sha256sum -c SHA256SUMS)
sudo install -d -o ubuntu -g ubuntu -m 0755 \
  /opt/gta-ai/27b/{bin,config} /opt/gta-ai/router/{bin,config,state} /opt/gta-ai/logs
sudo install -o root -g root -m 0755 "$ROOT/27b/bin/run-vllm" /opt/gta-ai/27b/bin/
sudo install -o root -g root -m 0644 "$ROOT/27b/config/vllm.env" /opt/gta-ai/27b/config/
sudo install -o root -g root -m 0755 "$ROOT/router/bin/"* /opt/gta-ai/router/bin/
sudo install -o root -g root -m 0644 "$ROOT/router/config/router.env" /opt/gta-ai/router/config/
SITE_PACKAGES="$(/opt/gta-ai/router/app/.venv/bin/python -c \
  'import site; print(site.getsitepackages()[0])')"
sudo install -d -o ubuntu -g ubuntu -m 0755 "$SITE_PACKAGES/gta_ai"
sudo install -o ubuntu -g ubuntu -m 0644 "$ROOT/router/src/gta_ai/"*.py \
  "$SITE_PACKAGES/gta_ai/"
sudo install -o root -g root -m 0644 "$ROOT/27b/systemd/gta-ai-vllm.service" /etc/systemd/system/
sudo install -o root -g root -m 0644 "$ROOT/router/systemd/gta-ai-router.service" /etc/systemd/system/
sudo systemctl daemon-reload
echo "27B/router已安装；未启动、未下载权重、未修改模型参数"
