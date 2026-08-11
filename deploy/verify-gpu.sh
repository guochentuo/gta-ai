#!/usr/bin/env bash
set -euo pipefail

readonly gta_ai_gpu_test_image="docker.io/library/ubuntu:24.04"

exec podman run --rm \
  --device nvidia.com/gpu=all \
  --security-opt=label=disable \
  "${gta_ai_gpu_test_image}" \
  nvidia-smi \
  --query-gpu=name,uuid,memory.total,driver_version \
  --format=csv,noheader

