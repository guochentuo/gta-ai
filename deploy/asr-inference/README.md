# GTA ASR 服务

使用 `faster-whisper-large-v3` 将 gta-worker 生成的 16 kHz 单声道 FLAC
转为带时间轴的 JSON、VTT 和纯文本。服务监听内网地址 `192.168.80.7:18084`，
供本机和测试节点共同调用，不暴露到公网。

服务按 30 秒窗口独立解码，默认不把前一窗口文字作为后一窗口的提示。
这样可避免旅游视频的持续背景音乐被 VAD 保留后，将一次误识别扩散为整段重复
字幕。可通过 `ASR_CHUNK_LENGTH_SEC` 和
`ASR_CONDITION_ON_PREVIOUS_TEXT` 调整，但生产环境应保持后者为 `false`。

每条音轨还会使用 25 秒窗口进行第二遍独立识别，比较两遍文字一致度、结尾覆盖和
重复片段比例。只有复核状态为 `verified` 才返回 `ready`；不一致时返回
`review_required`，worker 会写失败状态并进入既有重试/DLQ 链路。第二遍只保存校验
指标和文字 SHA-256，不把第二份文字混入最终 transcript。

接口：

```text
GET  /health
POST /v1/transcriptions
Content-Type: audio/flac
X-ASR-Language: auto
```

生产扩容时可以在多台推理节点部署，再通过 HAProxy/Keepalived 暴露统一地址；
gta-worker 只依赖 HTTP 接口，不依赖 Python 运行时。

GPU 推理依赖安装在 ASR 自己的虚拟环境中：

```bash
/opt/gta-ai/data/asr/.venv/bin/pip install 'nvidia-cublas-cu12>=12,<13' 'nvidia-cudnn-cu12>=9,<10'
```
