#!/usr/bin/env bash
set -euo pipefail

readonly gta_ai_vllm_image="docker.io/vllm/vllm-openai@sha256:7a0f0fdd2771464b6976625c2b2d5dd46f566aa00fbc53eceab86ef50883da90"

exec podman run --rm \
  --device nvidia.com/gpu=all \
  --security-opt=label=disable \
  --ipc=host \
  -v /opt/gta-ai/27b/models/Qwen3.8-27B-FP8:/models/current:ro \
  -v /opt/gta-ai/data/cache/huggingface:/root/.cache/huggingface:rw \
  -v /opt/gta-ai/data/cache/vllm:/root/.cache/vllm:rw \
  --entrypoint python3 \
  "${gta_ai_vllm_image}" \
  -c "import torch, vllm; print('vllm=' + vllm.__version__); print('torch=' + torch.__version__); print('torch_cuda=' + str(torch.version.cuda)); print('cuda_available=' + str(torch.cuda.is_available())); print('gpu=' + torch.cuda.get_device_name(0)); print('compute_capability=' + '.'.join(map(str, torch.cuda.get_device_capability(0))))"
