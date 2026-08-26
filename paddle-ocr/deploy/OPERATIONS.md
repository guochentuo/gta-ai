# PP-OCRv6 服务

开发节点`.2`在`http://192.168.80.2:7102`提供原始文字、坐标和置信度。接口为
`/v1/ocr`、`/v1/ocr-batch`和`/v1/risk-screen-batch`。模型只执行推理，不访问
Kafka、TiDB、ES或对象存储。

统一unit为`gta-ai-ocr.service`，复用`/opt/gta-ai/paddle-ocr/data`缓存并离线启动。
生产节点配置独立保存在`nodes/`。
