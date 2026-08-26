# ASR 推理服务

现有 `.7` 服务使用 `faster-whisper-large-v3`，地址为
`http://192.168.80.7:7104`。它只负责把 worker 提交的音频转换为带句级/词级时间轴的
结果，不消费 Kafka、不访问 TiDB/ES/Ceph，也不决定任务重试。

接口：`GET /health`、`POST /v1/transcriptions`。输入为 worker 从视频音轨提取的 16 kHz
单声道 FLAC。语音门禁判定无有效语音时，worker 可以不调用 ASR。

本轮保持 `.7` 现有 `gta-ai-asr.service`、虚拟环境、权重和参数，不由 `.2` 的统一发布包
安装或覆盖。启动和核验只允许：

```bash
sudo systemctl start gta-ai-asr.service
curl http://192.168.80.7:7104/health
```

ASR 结果属于私有分析事实，只能由 worker 持久化到 `dev-media-analysis`，不得进入播放桶、
OSS/CDN 或全球分发。
