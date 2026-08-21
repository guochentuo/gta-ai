# gta-ai

`gta-ai` 只提供无状态模型推理和27B准入路由：不消费业务Kafka、不决定任务、不选择Ceph路径、不写TiDB或ES。任务由`gta-worker`编排并持久化，结果由`gta-projection`投影。

## 模型职责、目录和开发地址

| 模型/服务 | `.2/.7`目录 | 地址与健康检查 | systemd | 设备与输出 |
|---|---|---|---|---|
| Qwen3-Embedding-0.6B | `.2 /opt/gta-ai/embedding` | `.2:18080/health` | `gta-ai-embedding-dev.service` | CPU，1024维文本向量 |
| PaddleOCR | `.2 /opt/gta-ai/ocr` | `.2:18081/health` | `gta-ai-ocr-dev.service` | CPU，文字、坐标、置信度 |
| SigLIP2 | `.2 /opt/gta-ai/siglip` | `.2:18082/health` | `gta-ai-image-embedding-dev.service` | CPU，768维视觉/文字同空间向量 |
| faster-whisper ASR | `.2 /opt/gta-ai/asr` | `.2:18084/health` | `gta-ai-asr-dev.service` | CPU int8，句级/词级时间轴 |
| AST AudioSet | `.2 /opt/gta-ai/ast` | `.2:18085/health` | `gta-ai-audio-events-dev.service` | CPU，环境声音标签；普通FEATURE默认不调用 |
| Qwen3.6-27B-FP8 vLLM | `.7 /opt/gta-ai/27b` | 后端仅`127.0.0.1:18086` | `gta-ai-vllm.service` | L40，图片/视频UNDERSTANDING |
| 27B Router | `.7 /opt/gta-ai/router` | `192.168.80.7:8000/health/live`、`/v1/models` | `gta-ai-router.service` | P2/P10准入、取消、超时和幽灵请求清理 |

模型权重和缓存位于各子目录`data/`；27B权重固定在`/opt/gta-ai/27b/models/Qwen3.6-27B-FP8`。发布包不包含权重，安装与启动必须使用既有缓存和离线参数，不得自动下载、替换revision或修改模型能力。

## 开发与生产拓扑

开发环境：

```text
.2: gta-worker + gta-projection + OCR/SigLIP/Qwen/ASR/AST(CPU)
.7: 27B vLLM + Router(L40)
worker基础模型调用 → 192.168.80.2:18080/18081/18082/18084/18085
worker 27B调用      → http://192.168.80.7:8000/v1/chat/completions
```

未来生产结构使用相同项目发布包：

```text
.7: 完整gta-worker + 27B Router/vLLM
.3/.4/.5/.6: gta-projection + OCR/SigLIP/Qwen/ASR/AST集群
```

基础模型通过HAProxy+Keepalived内网VIP接入。现有VIP约定为`192.168.80.130`，Qwen/OCR/SigLIP端口分别为18080/18081/18082；ASR/AST在生产启用集群前需按同一方式补齐18084/18085后端和健康检查。HAProxy必须限制`192.168.80.0/24`，Keepalived只在内网接口漂移VIP，不监听公网。worker生产配置只写VIP，不绑定具体模型节点；27B使用本机Router或明确内网地址，禁止绕过Router直接调用vLLM后端。

## 构建、打包和安装

```bash
cd /code/gta/gta-ai
./scripts/release.sh
sha256sum -c release/gta-ai-*.tar.gz.sha256

pkg=$(ls -1t release/gta-ai-*.tar.gz | head -1)
stage=$(mktemp -d)
tar -xzf "$pkg" -C "$stage"
cd "$stage/gta-ai"
sha256sum -c SHA256SUMS
sudo ./install-dev-package.sh       # 仅.2基础模型
sudo ./install-27b-package.sh       # 仅.7 27B/Router
rm -rf "$stage"
```

安装根只能是`/opt/gta-ai`，不会创建`gta-platform`总包或多层release运行目录，不移动或下载权重，也不自动启动服务。

## 启动、停止、重启和健康检查

`.2`：

```bash
sudo systemctl start gta-ai-embedding-dev.service gta-ai-ocr-dev.service \
  gta-ai-image-embedding-dev.service gta-ai-asr-dev.service \
  gta-ai-audio-events-dev.service
sudo systemctl stop gta-ai-embedding-dev.service gta-ai-ocr-dev.service \
  gta-ai-image-embedding-dev.service gta-ai-asr-dev.service \
  gta-ai-audio-events-dev.service
sudo systemctl restart gta-ai-embedding-dev.service gta-ai-ocr-dev.service \
  gta-ai-image-embedding-dev.service gta-ai-asr-dev.service \
  gta-ai-audio-events-dev.service
systemctl status 'gta-ai-*-dev.service' --no-pager
/opt/gta-ai/bin/verify-dev-models
```

`.7`：

```bash
sudo systemctl start gta-ai-vllm.service gta-ai-router.service
sudo systemctl stop gta-ai-router.service gta-ai-vllm.service
sudo systemctl restart gta-ai-router.service gta-ai-vllm.service
systemctl status gta-ai-vllm.service gta-ai-router.service --no-pager
curl -fsS http://192.168.80.7:8000/health/live
curl -fsS http://192.168.80.7:8000/v1/models
curl -fsS http://192.168.80.7:8000/_gta/runtime
```

## 日志、超时、取消和故障恢复

应用日志统一写`/opt/gta-ai/logs/`，UTF-8 JSON、UTC RFC3339毫秒时间、20MiB、保留10份。不得记录Token、图片Base64、完整向量或完整模型请求正文。

```bash
tail -f /opt/gta-ai/logs/gta.ai.ocr.dev.jsonl
tail -f /opt/gta-ai/logs/gta.ai.siglip.dev.jsonl
tail -f /opt/gta-ai/logs/gta.ai.embedding.dev.jsonl
tail -f /opt/gta-ai/logs/gta.ai.asr.dev.jsonl
tail -f /opt/gta-ai/logs/gta.ai.ast.dev.jsonl
tail -f /opt/gta-ai/logs/gta.ai.router.jsonl
tail -f /opt/gta-ai/logs/gta.ai.vllm.jsonl
```

worker请求必须带确定性requestId并设置有界超时；客户端超时或任务取消时调用Router取消接口。Router在断开、取消和异常的`finally`中释放waiting、active、workload和GPU准入，重试相同requestId保持幂等。Router故障时不得提交任务offset或写正式UNDERSTANDING标记；恢复后优先复用Ceph已完成结果。

## CPU、内存和GPU检查

```bash
# .2
systemd-cgtop --depth=3
free -h
vmstat 1 5
iostat -xz 1 5
df -h /opt /data
podman stats --no-stream

# .7
nvidia-smi
nvidia-smi dmon -s pucvmet -c 5
free -h
systemctl show gta-ai-vllm.service gta-ai-router.service \
  -p ActiveState -p MainPID -p NRestarts
```

故障恢复不重新下载权重：重新安装同一SHA发布包，核对既有`data/`、27B模型目录和Router虚拟环境，再逐个启动并执行健康检查。生产节点部署、VIP切换和负载均衡变更必须独立授权。
