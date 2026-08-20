# Qwen3-Embedding-0.6B

开发环境 `.2` 独立运行 `Qwen/Qwen3-Embedding-0.6B` CPU 服务：

```text
http://192.168.80.2:18080
GET /health
POST /v1/embeddings
```

模型 revision 固定在 dev env 中，输出 1024 维归一化文本向量。业务文本向量和内容理解
向量是不同字段，但使用相同模型；不得覆盖 SigLIP2 的 768 维视觉向量。

唯一部署来源是 `gta-worker/scripts/release.sh` 生成的 `gta-ai` 独立发布包，开发 unit 为
`gta-ai-embedding-dev.service`。服务复用已有 `/opt/gta-ai/embedding/data`，离线启动，不下载
模型。未来生产集群模板仍保留在源码中，本轮禁止部署或操作 `.3/.4/.5/.6`。
