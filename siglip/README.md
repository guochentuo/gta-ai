# 图片分析

本项目负责SigLIP2图片向量、文字向量和图片级分析。向量固定为768维，与业务文本使用的
1024维Qwen向量相互独立，禁止混用。

HTTP入口为`app.py`，运行服务为`gta-ai-image-embedding.service`。
