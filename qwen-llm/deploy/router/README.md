# 统一推理路由部署

这里保存统一推理路由的配置、启动脚本和systemd服务。路由负责流式转发、请求取消、
超时控制、GPU准入、优先级和共享GPU租约，不保存模型权重。启用业务知识检索后，
`/v1/chat/completions`会先调用Qwen Embedding生成1024维查询向量，再从Elasticsearch
业务索引检索资料并注入模型上下文；检索失败会自动降级为普通对话。

systemd服务名为`gta-ai-router.service`。
