# 大语言模型部署

这里保存Qwen3.8-27B-FP8的vLLM部署和可选身份LoRA训练配置。

- `vllm/`：模型推理服务。
- `training/`：身份数据、训练和LoRA生命周期。
- `runtime-versions.env`：GPU、CUDA、Podman、PyTorch和vLLM版本基线。
- `nvidia-cdi-refresh.env`：NVIDIA CDI运行配置。
- `verify-gpu.sh`：验证容器是否能够访问GPU。

27B专用Router位于`qwen-llm/router/`，部署文件位于`qwen-llm/deploy/router/`。
