# gta-ai

`gta-ai` 是独立的 Python 3.12 AI 分析与决策服务，与 `gta-worker` 解耦。当前主要包含：

- 本地 OpenAI 兼容模型客户端；
- OpenAI Responses API 最终决策客户端；
- 严格的 Pydantic 请求与结果结构；
- 健康检查、浏览器聊天页面和离线测试；
- Java 提交的异步图片/视频审核、ES 结果缓存和 HMAC 结果回调；
- Java 通过 `/v1/material-search` 使用的素材只读混合检索适配器；
- Qwen3.6 推理、LoRA 身份微调及推理优先的可抢占调度。

服务永远不连接 TiDB/MySQL，也不持有数据库账号。素材审核只允许访问 ES、Ceph、Kafka
和模型端点；只有 Java 能在签名回调事务中更新素材业务状态。

Java 不连接 ES。无关键词的素材管理列表直接读取 TiDB；关键词检索由 Java 调用
`POST /v1/material-search`，ES 凭据、索引名、Qwen 查询向量和查询 DSL 全部封装在
`gta-ai` 内。`gta-projection` 仍是素材 ES 投影的写入者。

## 素材审核接口

Java 使用 Bearer token 调用 `POST /v1/media-audits`，删除或重新提交时调用
`DELETE /v1/media-audits/{request_id}`。请求只携带素材 ID、代次、输入指纹和 ES/Ceph
定位信息，不携带数据库连接信息。

审核完成后，`gta-ai` 覆盖 `gta_media_content_audit` 中同一素材的 V1 文档，再调用 Java
的 `/internal/material-ai-audits/completed`。回调签名为：

```text
hex(HMAC-SHA256(callback_secret, timestamp + "\n" + raw_json_body))
```

Java 校验签名和五分钟时间窗后，才在本地事务内写 TiDB。回调失败不会伪造完成状态；Java
会把失联的 RUNNING 请求重新提交，相同输入优先命中 ES 缓存。

## 本地开发

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -U pip
.venv/bin/python -m pip install -e '.[dev]'
cp .env.example .env
.venv/bin/gta-ai
```

默认 API 监听 `127.0.0.1:8080`：

```bash
curl http://127.0.0.1:8080/health/live
curl http://127.0.0.1:8080/health/ready
```

`/health/live` 不调用外部服务；`/health/ready` 只探测配置的本地模型，并检查 OpenAI
密钥是否已配置，不会真正调用 OpenAI。

## 测试

```bash
.venv/bin/ruff check .
.venv/bin/pytest
```

单元测试使用进程内假服务或 HTTP Mock，不会调用真实模型提供商。

## 部署目录

- 源码与测试：`/code/gta/gta-ai`
- 运行部署：`/opt/gta-ai`
- 模型：`/opt/gta-ai/models`
- 运行数据：`/opt/gta-ai/data`
- 日志：`/opt/gta-ai/logs`
- 微调数据与适配器：`/opt/gta-ai/training`

## GPU 推理

宿主机使用 rootless Podman 和 NVIDIA CDI。运行版本及固定的 vLLM 镜像摘要记录在
`deploy/runtime-versions.env`。

```bash
/opt/gta-ai/bin/verify-gpu
/opt/gta-ai/bin/verify-vllm
systemctl --user status gta-ai-vllm.service
curl http://127.0.0.1:8000/v1/models
```

基础模型为 `/opt/gta-ai/models/Qwen3.6-27B-FP8`。vLLM 后端只监听
`127.0.0.1:18086`，稳定入口 `127.0.0.1:8000` 由推理路由提供；现有调用方不需要改端口。

## 空闲微调与请求抢占

身份知识使用 QLoRA 写入模型适配器权重，训练数据中没有 system 消息。运行规则如下：

1. 只有存在尚未部署、经过确认的训练版本，并且连续空闲 15 分钟时，才停止 vLLM 并开始训练；
2. 任意新的 `/v1/*` 推理请求到达后，路由先登记请求，调度器在 250 毫秒轮询周期内终止整组训练进程；
3. vLLM 恢复后，等待中的原请求继续执行；冷恢复耗时取决于模型加载速度；
4. 被抢占的训练保留 checkpoint，下次满足空闲条件后续训；
5. 新适配器必须通过无 system 消息的固定验收集，失败时不会切换线上权重；
6. 相同训练版本只执行一次，不会使用在线请求或模型回答自动自我训练。

状态与日志：

```bash
cat /opt/gta-ai/training/runtime/router-state.json
cat /opt/gta-ai/training/runtime/trainer-state.json
cat /opt/gta-ai/training/runtime/deployed.json
tail -f /opt/gta-ai/logs/identity-training-*.log
```

## 浏览器页面

浏览器服务默认监听 `8080`。它与 GPU 推理服务独立；推理停止时页面仍可打开，但模型恢复前不会返回答案。

```bash
systemctl --user start gta-ai-web.service
systemctl --user status gta-ai-web.service
tail -f /opt/gta-ai/logs/web.log
```

## 模型验收

可重复的模型验收程序位于 `acceptance/`。身份适配器另使用
`training/identity/evaluate_identity.py`，强制只发送单条 user 消息，防止把上下文注入误判成权重微调成功。
