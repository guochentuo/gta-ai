# Qwen3-Embedding-0.6B

`.2`独立运行`Qwen/Qwen3-Embedding-0.6B` CPU服务：

```text
http://192.168.80.2:7101
GET /health
POST /v1/embeddings
```

模型 revision 固定在 dev env 中，输出 1024 维归一化文本向量。业务文本向量和内容理解
向量是不同字段，但使用相同模型；不得覆盖 SigLIP2 的 768 维视觉向量。

部署来源是独立的`qwen-embedding/`项目，systemd unit为
`gta-ai-embedding.service`。模型数据保存在`/opt/gta-ai/qwen-embedding/data`，离线启动，
不得自动下载模型。
