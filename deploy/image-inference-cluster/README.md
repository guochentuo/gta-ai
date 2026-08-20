# OCR 与 SigLIP2 基础服务

开发节点 `.2` 运行两个相互独立的 CPU 推理服务：

| 服务 | 地址 | 输出 |
| --- | --- | --- |
| PaddleOCR | `http://192.168.80.2:18081` | 原始文字、坐标、置信度 |
| SigLIP2 | `http://192.168.80.2:18082` | 768 维图片/文字同空间向量 |

OCR 接口为 `/v1/ocr`、`/v1/ocr-batch` 和 `/v1/risk-screen-batch`；SigLIP2 接口为
`/v1/image-embeddings` 和 `/v1/text-embeddings`。模型只执行推理，不访问 Kafka、TiDB、ES
或对象存储。

开发 unit 为 `gta-ai-ocr-dev.service` 和 `gta-ai-image-embedding-dev.service`。它们由唯一
`gta-ai` 独立发布包安装，复用现有 `/opt/gta-ai/ocr/data` 与
`/opt/gta-ai/siglip/data` 缓存并离线启动。运行日志统一写 `/opt/gta-ai/logs/*.jsonl`。

Qwen 1024 维文本向量与 SigLIP2 768 维视觉向量不是同一空间，禁止混用。未来生产集群文件
只作为模板保留；本轮不得部署或操作 `.3/.4/.5/.6`。
