# Phase 4 acceptance harness

These scripts validate the local Qwen/vLLM deployment without connecting Google Ads, databases,
Redis or Elasticsearch and without changing `gta-worker`.

Run against an already-ready model service:

```bash
result_root=/opt/gta-ai/data/acceptance/manual-run
mkdir -p "$result_root/media" "$result_root/results"

acceptance/build_media.sh "$result_root/media"
.venv/bin/python acceptance/content_tests.py "$result_root/results"
.venv/bin/python acceptance/vision_tests.py "$result_root/media" "$result_root/results"
.venv/bin/python acceptance/long_context_test.py "$result_root/results"
.venv/bin/python acceptance/stress_test.py "$result_root/results" --requests 20
.venv/bin/python acceptance/tool_call_test.py "$result_root/results"
.venv/bin/python acceptance/service_cycle_test.py "$result_root/results"
```

The lifecycle test intentionally stops and restarts `gta-ai-vllm.service`. A complete run takes
several minutes because every start includes multimodal profiling. The NVENC coexistence test is
kept as an explicit operator step so it cannot accidentally start a hardware encoder during a
routine content-only acceptance run.
