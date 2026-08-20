#!/usr/bin/env bash
set -eu

exec /opt/gta-ai/router/app/.venv/bin/python -c \
  'from gta_ai.inference_router import main; main()'
