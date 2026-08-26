# AST 环境声音服务

AudioSet AST 只输出声音候选标签、置信度和时间范围，不消费任务、不写数据库或 ES，也不把
标签混入 ASR 文本。开发地址为 `http://192.168.80.2:7105`，健康检查为 `GET /health`，
推理接口为 `POST /v1/audio-events`。

`.2`使用统一发布包中的`gta-ai-audio-events.service`，复用已有容器镜像和
`/opt/gta-ai/ast-audioset/data`缓存，强制离线启动。当前自动媒体任务不启用AST；服务健康
常驻不代表会执行推理。
