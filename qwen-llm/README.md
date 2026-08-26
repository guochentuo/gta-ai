# 大语言模型

本项目负责Qwen3.8-27B-FP8推理、vLLM运行时、专用Router和可选LoRA训练，不查询
Elasticsearch，也不执行ASR、OCR或媒体任务编排。

运行服务：`gta-ai-vllm.service`和`gta-ai-router.service`。Router只管理本项目的27B
GPU请求，不代理其他模型。
