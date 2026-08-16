# GTA Embedding CPU 集群

该目录用于在 Elasticsearch 所在的三台服务器上部署独立的
`Qwen/Qwen3-Embedding-0.6B` CPU 推理服务。

## 拓扑

| 角色 | 地址 |
| --- | --- |
| Keepalived / HAProxy 入口 | `http://192.168.80.130:18080` |
| 单机测试环境 | `http://192.168.80.2:18080` |
| Embedding 节点 1 | `http://192.168.80.4:18080` |
| Embedding 节点 2 | `http://192.168.80.5:18080` |
| Embedding 节点 3 | `http://192.168.80.6:18080` |

Keepalived 继续复用现有 `VI_ES_WEB` 和 `192.168.80.130`，无需新增 VIP。
HAProxy 使用 HTTP 健康检查和 `leastconn`，自动排除故障节点。
整个服务只使用 `18080`，不占用现有的 `8000`、`8001`、`3000` 或 `3001`。

该 VIP 同时承载现有 Elasticsearch 的 HAProxy 入口，因此不要为了测试 Embedding
而直接停止 Keepalived。后端摘除可以通过单独停止一个 `gta-ai-embedding` 实例验证，
不会影响 Elasticsearch VIP。

`192.168.80.2` 是独立的单机测试环境，使用相同镜像、模型 revision 和接口契约，
但不加入正式环境的 HAProxy 后端。测试环境的 Java 服务应直接连接
`http://192.168.80.2:18080/v1/embeddings`。

## 资源边界

每个实例限制为 8 个 CPU、16GB 内存、64 个并发排队请求、每批最多 16 段文本和
4096 个 token。systemd 同时设置较低 CPU/IO 权重和较高 OOM 分值，避免影响同机
的 TiDB、Elasticsearch 和 Ceph。进入模型前应将正文切成约 500–1000 token 的
语义段落，不要把整篇长文章作为单个向量输入。

模型固定为 1024 维。ES 中对应字段必须显式配置为：

```json
{
  "type": "dense_vector",
  "dims": 1024,
  "index": true,
  "similarity": "cosine",
  "index_options": { "type": "int8_hnsw" }
}
```

## 接口

使用 OpenAI 兼容接口：

```bash
curl http://192.168.80.130:18080/v1/embeddings \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "Qwen/Qwen3-Embedding-0.6B",
    "input": ["杭州西湖旅游攻略"],
    "encoding_format": "float"
  }'
```

`gta-projection` 后续只连接 VIP，不直接绑定某个推理节点。文档向量与查询向量必须
使用同一个模型 revision；模型升级时应创建新的向量字段或版本化索引并重新回填。

健康检查地址为 `/health`。服务由 systemd 管理：

```bash
sudo systemctl status gta-ai-embedding
sudo journalctl -u gta-ai-embedding -n 100 --no-pager
```

首次启动需要下载约 1.2GB 模型；缓存命中后无需重新下载，但 CPU 模型仍需约两分钟
完成预热。预热期间 `/health` 不会返回 200，HAProxy 会自动保持该节点为不可用状态。

模型仓库 revision、容器镜像 digest、模型文件大小和 SHA-256 固定在
`model-versions.env` 中。升级任何一项时，都应先在新索引或新向量字段进行全量重建，
不能让同一向量字段混入不同模型版本的结果。

每台机器使用 `nodes/` 下与服务网 IP 对应的节点配置；不要把管理网
`192.168.70.x` 写入 `TEI_HOST`。
