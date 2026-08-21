#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
[[ -f "$ROOT/RELEASE-MANIFEST.json" && -x "$ROOT/bin/json-log-runner.py" ]] || {
  echo "请从解压后的gta-ai包执行" >&2; exit 2;
}
(cd "$ROOT" && sha256sum -c SHA256SUMS)
sudo install -d -o ubuntu -g ubuntu -m 0755 /opt/gta-ai/{bin,systemd,embedding,ocr,siglip,ast,asr}
sudo install -d -o ubuntu -g ubuntu -m 0750 /opt/gta-ai/logs
for model in embedding ocr siglip ast asr; do
  sudo install -d -o ubuntu -g ubuntu -m 0755 "/opt/gta-ai/$model/config"
  sudo install -d -o ubuntu -g ubuntu -m 0750 "/opt/gta-ai/$model/data"
  sudo find "/opt/gta-ai/$model/config" -mindepth 1 -maxdepth 1 -type f -delete
done
sudo find /opt/gta-ai/bin /opt/gta-ai/systemd -mindepth 1 -maxdepth 1 -type f -delete
sudo install -o ubuntu -g ubuntu -m 0755 "$ROOT"/bin/* /opt/gta-ai/bin/
for model in embedding ocr siglip ast asr; do
  sudo install -o root -g ubuntu -m 0640 "$ROOT/$model/config"/* "/opt/gta-ai/$model/config/"
done
sudo install -o ubuntu -g ubuntu -m 0644 "$ROOT/asr/app.py" /opt/gta-ai/asr/app.py
sudo install -o root -g root -m 0644 "$ROOT"/systemd/*.service /opt/gta-ai/systemd/
sudo install -o root -g root -m 0644 "$ROOT"/systemd/*.service /etc/systemd/system/
sudo install -o ubuntu -g ubuntu -m 0644 "$ROOT/README.md" "$ROOT/RELEASE-MANIFEST.json" "$ROOT/SHA256SUMS" /opt/gta-ai/
sudo systemctl daemon-reload
echo "已安装到/opt/gta-ai；未移动模型权重，也未自动启动模型"
