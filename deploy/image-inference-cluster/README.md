# GTA 图片理解基础服务集群

本目录部署两个相互独立的CPU推理服务：

- `gta-ai-ocr`：使用PaddleOCR提取文字、置信度和位置；
- `gta-ai-image-embedding`：使用SigLIP2生成图片向量及同空间文字向量。

生产入口分别为`192.168.80.130:18081`和`192.168.80.130:18082`。`.3/.5/.6`
为活动节点，`.4`为HAProxy热备。`.2`部署相同版本的单机测试服务，但不加入生产HAProxy。

模型缓存在每台节点本地`/opt`目录，服务运行不依赖`192.168.80.7:/data/gta`的NFS。
NFS只用于分发经过校验的容器归档和模型缓存，避免NFS或`.7`故障导致整个推理集群无法重启。

`VI_ES_WEB`通过`gta_haproxy_es_web`脚本跟踪本机HAProxy状态。连续三次失败后本机
VRRP优先级降低60，避免VIP停留在HAProxy已故障的节点。

OCR接口：

```bash
curl http://192.168.80.130:18081/v1/ocr -F file=@image.jpg
```

密集审核帧使用批量接口，单批默认最多32张；调用方可并发提交多个批次，让HAProxy把
请求分散到不同OCR节点：

```bash
curl http://192.168.80.130:18081/v1/ocr-batch \
  -F files=@frame-001.jpg \
  -F files=@frame-002.jpg
```

图片向量接口：

```bash
curl http://192.168.80.130:18082/v1/image-embeddings -F files=@image.jpg
```

视频关键帧可以同时提交与图片一一对应的归一化烧录字幕框。服务不会改写原图或生成
派生图片，而是在 SigLIP2 的 `pixel_attention_mask` 中把这些框覆盖的 patch 置为不可见：

```bash
curl http://192.168.80.130:18082/v1/image-embeddings \
  -F files=@image.jpg \
  -F 'subtitle_masks=[[[0.20,0.78,0.80,0.88]]]'
```

响应中的 `input_policy`、`embedding_input_sha256` 和 `subtitle_patch_mask` 用来审计本次
向量是否真正应用了字幕隔离。牌匾、店招、Logo、景点名称等非字幕区域不会被屏蔽。

同向量空间的文字查询接口：

```bash
curl http://192.168.80.130:18082/v1/text-embeddings \
  -H 'Content-Type: application/json' \
  -d '{"input":["乌镇夜景"]}'
```

图片向量与现有Qwen3文本向量不是同一个向量空间，不能交叉使用。ES必须为SigLIP2
建立独立的768维向量字段，并记录模型revision、维度、归一化方式和图片内容hash。
