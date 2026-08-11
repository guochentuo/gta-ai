# gta-ai

Independent Python 3.12 service for local AI analysis and OpenAI-assisted final decisions.

This repository is intentionally isolated from `gta-worker` and currently contains only:

- environment-based configuration;
- an OpenAI-compatible local model client;
- an OpenAI Responses API final-decision client;
- strict Pydantic request and decision schemas;
- liveness and readiness endpoints;
- offline unit tests with mocked model clients.

Google Ads, databases, Redis, Elasticsearch, Kafka and Ceph are deliberately not connected in
this phase.

The root page is a self-contained browser chat interface. It supports streamed responses,
thinking mode, multiple image uploads, and browser-local conversation history. The browser calls
`/api/chat`; that endpoint proxies requests to the loopback-only vLLM service, so port 8000 never
needs to be exposed to the LAN.

## Development

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -U pip
.venv/bin/python -m pip install -e '.[dev]'
cp .env.example .env
.venv/bin/gta-ai
```

The default API listens on `127.0.0.1:8080`.

```bash
curl http://127.0.0.1:8080/health/live
curl http://127.0.0.1:8080/health/ready
```

`/health/live` never calls an external service. `/health/ready` probes only the configured local
model endpoint and reports whether the OpenAI key is configured; it does not call OpenAI.

## Tests

```bash
.venv/bin/ruff check .
.venv/bin/pytest
```

All tests use in-process fakes or HTTP mock transports and do not contact model providers.

## Deployment boundary

- Source and tests: `/code/gta/gta-ai`
- Release deployment: `/opt/gta-ai`
- Model files: `/opt/gta-ai/models`
- Runtime data: `/opt/gta-ai/data`
- Runtime logs: `/opt/gta-ai/logs`

## GPU inference runtime

The host uses rootless Podman with NVIDIA CDI. The installed runtime versions and pinned vLLM
image digest are recorded in `deploy/runtime-versions.env`.

```bash
/opt/gta-ai/bin/verify-gpu
/opt/gta-ai/bin/verify-vllm
```

Persistent paths are mounted explicitly from `/opt/gta-ai`:

- model weights: `/opt/gta-ai/models/Qwen3.6-27B-FP8`;
- Hugging Face cache: `/opt/gta-ai/data/cache/huggingface`;
- vLLM compile cache: `/opt/gta-ai/data/cache/vllm`;
- future Qdrant storage: `/opt/gta-ai/data/qdrant`.

The first model-server profile uses a 32K context window, one sequence, FP8 KV cache, and an
initial GPU memory utilization limit of 0.82. Runtime files are deployed under `/opt/gta-ai`:

```bash
/opt/gta-ai/bin/run-vllm
systemctl --user status gta-ai-vllm.service
curl http://127.0.0.1:8000/v1/models
```

The model server listens only on `127.0.0.1:8000` and exposes the OpenAI-compatible API.

The user service is intentionally started on demand instead of enabled at boot, so training can
own the full GPU when inference is not needed:

```bash
systemctl --user start gta-ai-vllm.service
systemctl --user stop gta-ai-vllm.service
systemctl --user status gta-ai-vllm.service
tail -f /opt/gta-ai/logs/vllm.log
```

A full cold start includes multimodal memory profiling and takes about four minutes on this host.
The local model client timeout is 600 seconds so a 2K-token thinking response is not cut off at the
HTTP layer.

## Browser interface

The deployed Web service listens on port 8080. It is independent of the GPU-backed vLLM service:
the page remains available while inference is stopped, and its status indicator reports when the
model becomes ready.

```bash
systemctl --user status gta-ai-web.service
systemctl --user restart gta-ai-web.service
tail -f /opt/gta-ai/logs/web.log
```

## Model acceptance

The reproducible phase-4 harness is in `acceptance/`. The completed server run is stored at
`/opt/gta-ai/data/acceptance/phase4-20260808T105232Z/REPORT.md`. It covers structured Chinese
analysis, article review and generation, multimodal inputs, tool calling, a 31K-token prompt,
continuous requests, NVENC coexistence and service lifecycle behavior.
