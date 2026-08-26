# 文本向量

本项目负责Qwen3-Embedding-0.6B和Elasticsearch业务向量使用的1024维文本向量协议，
但不负责向Elasticsearch写入业务文档。

HTTP API由TEI容器提供，接口为`POST /v1/embeddings`。配置位于`config/`，部署文件位于
`deploy/`，运行服务为`gta-ai-embedding.service`。

VS Code调试项`GTA AI: Qwen Embedding`使用`config/debug.env`在
`192.168.80.7:7101`启动容器并将TEI日志输出到集成终端。TEI是预编译Rust服务，该模式
用于观察请求和运行日志，不支持Python断点。
