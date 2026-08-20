# gta-ai

`gta-ai` 只提供无状态推理，不调度任务、不决定 Ceph 路径、不写 TiDB/ES。`.2` 部署
OCR、SigLIP2、Qwen3-Embedding-0.6B、AST 与 CPU faster-whisper；`.7` 只保留 27B 和
router。模型结果由 `gta-worker` 持久化，再由 `gta-projection` 投影。

| 子目录 | 地址 | 设备 | 输出 |
| --- | --- | --- | --- |
| `/opt/gta-ai/embedding` | `.2:18080` | CPU | Qwen 1024维文本向量 |
| `/opt/gta-ai/ocr` | `.2:18081` | CPU | OCR文字/坐标/置信度 |
| `/opt/gta-ai/siglip` | `.2:18082` | CPU | SigLIP2 768维视觉向量 |
| `/opt/gta-ai/asr` | `.2:18084` | CPU int8 | Whisper转写与时间轴 |
| `/opt/gta-ai/ast` | `.2:18085` | CPU | 环境声音；审核未启用 |
| `/opt/gta-ai/27b`、`/opt/gta-ai/router` | `.7:8000` | L40 | UNDERSTANDING；常驻 |

`.7` 的运行内容只允许位于 `/opt/gta-ai/27b`、`/opt/gta-ai/router`，应用日志位于
`/opt/gta-ai/logs`。27B 的脚本、配置、模型、缓存与适配器均归入 `27b/`；router 的虚拟
环境、配置和准入状态均归入 `router/`。ASR 不再部署在 `.7`。

## 唯一打包和安装

```bash
cd /code/gta/gta-ai
./scripts/release.sh
sha256sum -c release/gta-ai-*.tar.gz.sha256
mkdir -p /tmp/gta-ai-install
tar -xzf release/gta-ai-*.tar.gz -C /tmp/gta-ai-install
/tmp/gta-ai-install/gta-ai/install-dev-package.sh
```

包只含代码、dev配置、unit、manifest与校验，不复制权重、不下载模型。安装根固定为
`/opt/gta-ai`，每个模型有独立 `config/` 和 `data/`；日志统一在 `/opt/gta-ai/logs`。
禁止总平台包、release软链接或源码目录启动。

同一个已校验项目包在 `.7` 使用 27B/router 安装入口；它只安装代码、配置和unit，不下载
或复制权重，也不自动启动服务：

```bash
/tmp/gta-ai-install/gta-ai/install-27b-package.sh
```

```bash
sudo cp /opt/gta-ai/systemd/*.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl start gta-ai-embedding-dev.service gta-ai-ocr-dev.service \
  gta-ai-image-embedding-dev.service gta-ai-asr-dev.service gta-ai-audio-events-dev.service
sudo systemctl stop gta-ai-asr-dev.service
sudo systemctl restart gta-ai-asr-dev.service
systemctl status 'gta-ai-*-dev.service' --no-pager
/opt/gta-ai/bin/verify-dev-models
tail -f /opt/gta-ai/logs/gta.ai.asr.dev.jsonl
```

`.7` 的 27B/router 只允许使用以下命令；vLLM 后端仅监听
`127.0.0.1:18086`，router 仅监听内网 `192.168.80.7:8000`：

```bash
sudo systemctl start gta-ai-vllm.service gta-ai-router.service
sudo systemctl stop gta-ai-router.service gta-ai-vllm.service
sudo systemctl restart gta-ai-vllm.service gta-ai-router.service
systemctl status gta-ai-vllm.service gta-ai-router.service --no-pager
curl -fsS http://192.168.80.7:8000/health/live
curl -fsS http://192.168.80.7:8000/v1/models
tail -f /opt/gta-ai/logs/gta.ai.router.jsonl
tail -f /opt/gta-ai/logs/gta.ai.vllm.jsonl
```

日志是一行一个 UTF-8 JSON，时间为 UTC RFC3339 毫秒，20 MiB 轮转、10 个备份；journal
仅保留 systemd 生命周期。所有模型强制使用现有缓存和离线参数；不得自动下载或更改模型
revision/能力。

## 备份、恢复和验证

本安装器不自动备份，也不覆盖各子目录 `data/`。显式备份仅在相应模型停止时执行：

```bash
sudo tar -C /opt/gta-ai -czf /tmp/gta-ai-config.tgz \
  embedding/config ocr/config siglip/config asr/config ast/config
sudo tar -C /opt/gta-ai -xzf /tmp/gta-ai-config.tgz
```

恢复后先检查配置与缓存目录，再逐个启动并执行 `/health`。代码/脚本只能通过同一项目发布
包恢复，不能从临时目录恢复。源码测试使用 `pytest`，测试产物不进入发布包或运行目录。
