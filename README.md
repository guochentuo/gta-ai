# gta-ai

本仓库由六个相互独立的模型项目组成。每个模型自己维护HTTP接口、
配置和部署文件，不共享统一模型进程，也不再使用`inference/`总目录。

## 目录结构

```text
gta-ai/
├── qwen-llm/            # Qwen大语言模型、专用Router和训练工具
├── qwen-embedding/      # Qwen文本向量模型
├── faster-whisper/      # Faster Whisper语音识别
├── paddle-ocr/          # PaddleOCR文字识别
├── siglip/              # SigLIP图片分析
└── ast-audioset/        # AST AudioSet音频分析
```

每个模型项目可以独立复制、配置、启动、停止和升级。模型之间只允许通过HTTP协作，禁止
跨项目导入代码。

## 服务端口

| 项目 | 内部端口 | 接口 |
|---|---:|---|
| `qwen-llm/router` | 7100 | `/v1/chat/completions`、`/health/live` |
| `qwen-embedding` | 7101 | `/v1/embeddings`、`/health` |
| `paddle-ocr` | 7102 | `/v1/ocr`、`/health` |
| `siglip` | 7103 | `/v1/image-embeddings`、`/health` |
| `faster-whisper` | 7104 | `/v1/transcriptions`、`/health` |
| `ast-audioset` | 7105 | `/v1/audio-events`、`/health` |
| `qwen-llm/vLLM` | 7106 | 只允许本机Router访问 |
| `qwen-llm/tests/web` | 8080 | 27B测试页面 |

27B Router和vLLM只部署在GPU节点`.7`。其他CPU模型按各自项目独立部署。调用方直接访问
对应模型的HTTP接口；27B统一通过`192.168.80.7:7100`访问Router，不直接访问vLLM。

## 配置与部署

每个模型的运行配置位于自己的`config/`，systemd和启动脚本位于自己的`deploy/`：

```text
<model>/
├── app.py或router/app.py
├── config/
├── deploy/
└── README.md
```

Embedding和OCR将运行日志写入各自的logs/service.log，由deploy/logrotate.conf限制保留量；其他CPU模型使用journald。27B Router和vLLM
使用`qwen-llm/router/runtime_logging.py`统一分流到简洁控制台、本地滚动文件和Kafka。
GPU运行时版本、CDI配置和GPU验证脚本归属`qwen-llm/deploy/`。模型权重和缓存不进入Git，部署时放在
对应的`/opt/gta-ai/<项目名>/data`；27B权重继续使用
`/opt/gta-ai/27b/models/Qwen3.8-27B-FP8`。

旧Embedding、OCR和图片分析HAProxy集群已经撤除。这三个项目重新部署前必须填写新的节点
配置，不能继续使用旧VIP模型端口。ES的`192.168.80.130:9200`不受影响。

## 本地验证

```bash
cd /code/gta/gta-ai
PYTHONDONTWRITEBYTECODE=1 .venv/bin/pytest
.venv/bin/ruff check qwen-llm qwen-embedding faster-whisper paddle-ocr siglip ast-audioset
```

启动测试页面：

```bash
.venv/bin/python qwen-llm/tests/web/run.py
```

本次重构只调整代码和未来部署结构，不自动部署、启动或重启模型服务。
