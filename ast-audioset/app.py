from __future__ import annotations

import json
import os
import subprocess
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import numpy as np
import torch
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

SERVICE_NAME = "gta-ai-audio-events"
SERVICE_VERSION = "1.0.0"
MODEL_NAME = os.getenv(
    "AUDIO_EVENT_MODEL", "MIT/ast-finetuned-audioset-10-10-0.4593"
)
MODEL_CACHE = os.getenv("AUDIO_EVENT_MODEL_CACHE", "/opt/gta-ai/data/cache/audio-events")
DEVICE = os.getenv("AUDIO_EVENT_DEVICE", "cpu").strip().lower()
MAX_AUDIO_BYTES = int(os.getenv("AUDIO_EVENT_MAX_AUDIO_BYTES", str(256 * 1024 * 1024)))

_extractor: AutoFeatureExtractor | None = None
_model: AutoModelForAudioClassification | None = None
_model_lock = threading.Lock()
_inference_lock = threading.Lock()


def get_model() -> tuple[AutoFeatureExtractor, AutoModelForAudioClassification]:
    global _extractor, _model
    if _extractor is not None and _model is not None:
        return _extractor, _model
    with _model_lock:
        if _extractor is None or _model is None:
            _extractor = AutoFeatureExtractor.from_pretrained(
                MODEL_NAME, cache_dir=MODEL_CACHE, local_files_only=True
            )
            _model = AutoModelForAudioClassification.from_pretrained(
                MODEL_NAME, cache_dir=MODEL_CACHE, local_files_only=True
            )
            _model.eval()
            if DEVICE == "cuda":
                _model.cuda()
    return _extractor, _model


def classify(
    audio_bytes: bytes, window_sec: int, minimum_score: float, maximum_labels: int
) -> dict[str, Any]:
    started = time.perf_counter()
    sample_rate = 16_000
    decoded = subprocess.run(
        [
            "/usr/bin/ffmpeg", "-hide_banner", "-nostdin", "-loglevel", "error",
            "-i", "pipe:0", "-vn", "-ac", "1", "-ar", str(sample_rate),
            "-f", "f32le", "pipe:1",
        ],
        input=audio_bytes,
        capture_output=True,
        check=False,
        timeout=300,
    )
    if decoded.returncode != 0:
        raise ValueError(
            "ffmpeg音频解码失败: " + decoded.stderr.decode("utf-8", errors="replace")[-1000:]
        )
    waveform = np.frombuffer(decoded.stdout, dtype="<f4")
    duration = float(len(waveform)) / float(sample_rate) if sample_rate else 0.0
    if duration <= 0.0:
        raise ValueError("音频解码后为空")
    extractor, model = get_model()
    samples_per_window = max(sample_rate, int(window_sec * sample_rate))
    windows: list[dict[str, Any]] = []
    with _inference_lock, torch.inference_mode():
        for begin in range(0, len(waveform), samples_per_window):
            chunk = waveform[begin : begin + samples_per_window]
            if len(chunk) < sample_rate // 2:
                continue
            inputs = extractor(
                np.asarray(chunk, dtype=np.float32),
                sampling_rate=sample_rate,
                return_tensors="pt",
            )
            if DEVICE == "cuda":
                inputs = {key: value.cuda() for key, value in inputs.items()}
            logits = model(**inputs).logits[0]
            probabilities = torch.sigmoid(logits).detach().cpu()
            top = torch.topk(probabilities, k=min(maximum_labels, len(probabilities)))
            events = []
            for score, label_id in zip(
                top.values.tolist(), top.indices.tolist(), strict=True
            ):
                if float(score) < minimum_score:
                    continue
                events.append(
                    {
                        "label": str(model.config.id2label[int(label_id)]),
                        "score": round(float(score), 6),
                    }
                )
            windows.append(
                {
                    "startSec": round(begin / sample_rate, 3),
                    "endSec": round(
                        min(len(waveform), begin + samples_per_window) / sample_rate, 3
                    ),
                    "events": events,
                }
            )
    return {
        "schemaVersion": 1,
        "status": "ready",
        "model": MODEL_NAME,
        "sampleRate": sample_rate,
        "durationSec": round(duration, 3),
        "windowSec": window_sec,
        "minimumScore": minimum_score,
        "windows": windows,
        "elapsedMs": round((time.perf_counter() - started) * 1000, 3),
    }


class Handler(BaseHTTPRequestHandler):
    server_version = f"{SERVICE_NAME}/{SERVICE_VERSION}"

    def log_message(self, fmt: str, *args: object) -> None:
        print(f"[{self.log_date_time_string()}] {fmt % args}", flush=True)

    def send_json(self, status: HTTPStatus, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode()
        self.send_response(status.value)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path != "/health":
            self.send_json(HTTPStatus.NOT_FOUND, {"detail": "not found"})
            return
        self.send_json(
            HTTPStatus.OK,
            {
                "status": "ok",
                "service": SERVICE_NAME,
                "version": SERVICE_VERSION,
                "model": MODEL_NAME,
                "modelLoaded": _model is not None,
                "device": DEVICE,
            },
        )

    def do_POST(self) -> None:
        if self.path.split("?", 1)[0] != "/v1/audio-events":
            self.send_json(HTTPStatus.NOT_FOUND, {"detail": "not found"})
            return
        try:
            size = int(self.headers.get("Content-Length", "0"))
            window_sec = int(self.headers.get("X-Window-Seconds", "10"))
            minimum_score = float(self.headers.get("X-Minimum-Score", "0.25"))
            maximum_labels = int(self.headers.get("X-Maximum-Labels", "5"))
        except ValueError:
            self.send_json(HTTPStatus.BAD_REQUEST, {"detail": "请求参数非法"})
            return
        requested_model = self.headers.get("X-Audio-Model", MODEL_NAME)
        if requested_model != MODEL_NAME:
            self.send_json(HTTPStatus.CONFLICT, {"detail": "请求模型与固定模型不一致"})
            return
        if size <= 0 or size > MAX_AUDIO_BYTES or not (1 <= window_sec <= 60):
            self.send_json(HTTPStatus.BAD_REQUEST, {"detail": "音频大小或窗口参数非法"})
            return
        try:
            self.send_json(
                HTTPStatus.OK,
                classify(self.rfile.read(size), window_sec, minimum_score, maximum_labels),
            )
        except Exception as exception:
            self.send_json(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                {"detail": f"环境声音推理失败: {exception}"},
            )


def main() -> None:
    host = os.getenv("AUDIO_EVENT_HOST", "0.0.0.0")
    port = int(os.getenv("AUDIO_EVENT_PORT", "7105"))
    ThreadingHTTPServer((host, port), Handler).serve_forever()


if __name__ == "__main__":
    main()
