#!/usr/bin/env bash
set -eu

destination=/opt/gta-ai/qwen-llm/models/Qwen3-4B-Instruct-2507-Q4_K_M.gguf
url=https://huggingface.co/unsloth/Qwen3-4B-Instruct-2507-GGUF/resolve/main/Qwen3-4B-Instruct-2507-Q4_K_M.gguf

install -d -m 0755 "$(dirname "$destination")"
curl --fail --location --retry 5 --continue-at - --output "$destination.part" "$url"
mv "$destination.part" "$destination"
chmod 0644 "$destination"
