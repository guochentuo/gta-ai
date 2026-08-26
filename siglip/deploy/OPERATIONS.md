# SigLIP2 图片分析服务

开发节点 `.2` 在 `http://192.168.80.2:7103` 提供768维图片/文字同空间向量。
接口为 `/v1/image-embeddings` 和 `/v1/text-embeddings`。模型只执行推理，不访问
Kafka、TiDB、ES或对象存储。统一unit为`gta-ai-image-embedding.service`，复用
`/opt/gta-ai/siglip/data`缓存并离线启动。

Qwen 1024 维文本向量与 SigLIP2 768 维视觉向量不是同一空间，禁止混用。未来生产集群文件
只作为模板保留；生产操作必须单独授权。
