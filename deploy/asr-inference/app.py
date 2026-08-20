from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import os
import tempfile
import threading
import time
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
from collections import deque
from contextlib import contextmanager, suppress
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from faster_whisper import WhisperModel
from faster_whisper.audio import decode_audio
from faster_whisper.vad import VadOptions, get_speech_timestamps

SERVICE_NAME = "gta-ai-asr"
SERVICE_VERSION = "1.3.0"
MODEL_NAME = os.getenv("ASR_MODEL", "Systran/faster-whisper-large-v3")
MODEL_CACHE = os.getenv("ASR_MODEL_CACHE", "/opt/gta-ai/data/cache/faster-whisper")
DEVICE = os.getenv("ASR_DEVICE", "cuda")
COMPUTE_TYPE = os.getenv("ASR_COMPUTE_TYPE", "int8_float16")
BEAM_SIZE = int(os.getenv("ASR_BEAM_SIZE", "5"))
CHUNK_LENGTH_SEC = int(os.getenv("ASR_CHUNK_LENGTH_SEC", "30"))
VERIFY_ENABLED = os.getenv("ASR_VERIFY_ENABLED", "true").strip().lower() in (
    "1", "true", "yes", "on"
)
VERIFY_CHUNK_LENGTH_SEC = int(os.getenv("ASR_VERIFY_CHUNK_LENGTH_SEC", "25"))
VERIFY_MIN_TEXT_AGREEMENT = float(os.getenv("ASR_VERIFY_MIN_TEXT_AGREEMENT", "0.72"))
VERIFY_MAX_END_DIFFERENCE_SEC = float(
    os.getenv("ASR_VERIFY_MAX_END_DIFFERENCE_SEC", "15")
)
VERIFY_MIN_SPEECH_DURATION_AGREEMENT = float(
    os.getenv("ASR_VERIFY_MIN_SPEECH_DURATION_AGREEMENT", "0.90")
)
CONDITION_ON_PREVIOUS_TEXT = os.getenv(
    "ASR_CONDITION_ON_PREVIOUS_TEXT", "false"
).strip().lower() in ("1", "true", "yes", "on")
MAX_AUDIO_BYTES = int(os.getenv("ASR_MAX_AUDIO_BYTES", str(256 * 1024 * 1024)))
SPEECH_GATE_MINIMUM_SEC = float(os.getenv("ASR_SPEECH_GATE_MINIMUM_SEC", "0.8"))
GPU_ADMISSION_ENABLED = os.getenv("ASR_GPU_ADMISSION_ENABLED", "true").lower() == "true"
GPU_ADMISSION_ENDPOINT = os.getenv(
    "ASR_GPU_ADMISSION_ENDPOINT", "http://192.168.80.7:8000/_gta/admission"
).rstrip("/")
GPU_ADMISSION_MEMORY_MB = int(os.getenv("ASR_GPU_ADMISSION_MEMORY_MB", "6144"))
GPU_ADMISSION_WAIT_SECONDS = int(os.getenv("ASR_GPU_ADMISSION_WAIT_SECONDS", "1800"))
GPU_ADMISSION_TTL_SECONDS = int(os.getenv("ASR_GPU_ADMISSION_TTL_SECONDS", "1800"))

_model: WhisperModel | None = None
_model_lock = threading.Lock()


class PriorityInferenceGate:
    """单 GPU 串行推理门禁; P0/P1 始终排在尚未开始的 P9 前面。"""

    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._active = False
        self._realtime: deque[object] = deque()
        self._historical: deque[object] = deque()

    @contextmanager
    def acquire(self, priority: str):
        token = object()
        historical = priority.strip().upper() == "P9"
        queue = self._historical if historical else self._realtime
        with self._condition:
            queue.append(token)
            while self._active or queue[0] is not token or (
                historical and self._realtime
            ):
                self._condition.wait()
            queue.popleft()
            self._active = True
        try:
            yield
        finally:
            with self._condition:
                self._active = False
                self._condition.notify_all()

    def snapshot(self) -> dict[str, int | bool]:
        with self._condition:
            return {
                "active": self._active,
                "realtimeWaiting": len(self._realtime),
                "historicalWaiting": len(self._historical),
            }


_inference_gate = PriorityInferenceGate()


class GpuAdmissionClient:
    def __init__(self, endpoint: str) -> None:
        self.endpoint = endpoint

    def _json_request(self, request: urllib.request.Request) -> dict[str, Any]:
        with urllib.request.urlopen(request, timeout=GPU_ADMISSION_WAIT_SECONDS + 10) as response:
            return json.loads(response.read().decode("utf-8"))

    @contextmanager
    def lease(self, priority: str):
        if not GPU_ADMISSION_ENABLED or DEVICE != "cuda":
            yield
            return
        historical = priority.strip().upper() == "P9"
        owner = f"asr:{os.getpid()}:{threading.get_ident()}:{time.time_ns()}"
        body = json.dumps(
            {
                "owner": owner,
                "workload_class": (
                    "HISTORICAL_FEATURE" if historical else "REALTIME_FEATURE"
                ),
                "requested_memory_mb": GPU_ADMISSION_MEMORY_MB,
                "ttl_seconds": GPU_ADMISSION_TTL_SECONDS,
                "wait_seconds": GPU_ADMISSION_WAIT_SECONDS,
            }
        ).encode("utf-8")
        request = urllib.request.Request(
            f"{self.endpoint}/leases",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        lease = self._json_request(request)
        lease_id = str(lease["lease_id"])
        stopping = threading.Event()

        def heartbeat() -> None:
            interval = max(5, GPU_ADMISSION_TTL_SECONDS // 3)
            while not stopping.wait(interval):
                heartbeat_request = urllib.request.Request(
                    f"{self.endpoint}/leases/{lease_id}/heartbeat",
                    data=json.dumps(
                        {"owner": owner, "ttl_seconds": GPU_ADMISSION_TTL_SECONDS}
                    ).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with suppress(OSError, ValueError, urllib.error.URLError):
                    self._json_request(heartbeat_request)
                    # The active inference remains safe until the original TTL.
                    # A missing heartbeat is visible in admission state and logs.

        heartbeat_thread = threading.Thread(target=heartbeat, daemon=True)
        heartbeat_thread.start()
        try:
            yield
        finally:
            stopping.set()
            heartbeat_thread.join(timeout=2)
            query = urllib.parse.urlencode({"owner": owner})
            release_request = urllib.request.Request(
                f"{self.endpoint}/leases/{lease_id}?{query}", method="DELETE"
            )
            with suppress(OSError, ValueError, urllib.error.URLError):
                self._json_request(release_request)


_admission_client = GpuAdmissionClient(GPU_ADMISSION_ENDPOINT)


def get_model() -> WhisperModel:
    global _model
    if _model is not None:
        return _model
    with _model_lock:
        if _model is None:
            _model = WhisperModel(
                MODEL_NAME,
                device=DEVICE,
                compute_type=COMPUTE_TYPE,
                download_root=MODEL_CACHE,
                local_files_only=True,
            )
    return _model


def decode(audio_path: str, language: str | None, chunk_length_sec: int) -> dict[str, Any]:
    segments_iter, info = get_model().transcribe(
        audio_path,
        language=language,
        beam_size=BEAM_SIZE,
        word_timestamps=True,
        vad_filter=True,
        # 每个窗口独立解码。旅游视频中持续背景音乐会让 VAD 认为整条音轨
        # 都是语音; 继承前一窗口文本时, 一次错误很容易扩散成后续重复幻觉。
        condition_on_previous_text=CONDITION_ON_PREVIOUS_TEXT,
        chunk_length=chunk_length_sec,
    )
    segments = []
    texts = []
    for segment in segments_iter:
        text = segment.text.strip()
        if not text:
            continue
        texts.append(text)
        words = []
        for word in segment.words or []:
            words.append(
                {
                    "start": round(float(word.start), 3),
                    "end": round(float(word.end), 3),
                    "word": word.word,
                    "probability": round(float(word.probability), 4),
                }
            )
        segments.append(
            {
                "start": round(float(segment.start), 3),
                "end": round(float(segment.end), 3),
                "text": text,
                "words": words,
            }
        )
    return {
        "status": "ready" if segments else "no_speech",
        "language": info.language,
        "languageProbability": round(float(info.language_probability), 6),
        "durationSec": round(float(info.duration), 3),
        "durationAfterVadSec": round(float(info.duration_after_vad), 3),
        "text": "\n".join(texts),
        "segments": segments,
    }


def speech_gate(audio_path: str) -> dict[str, Any]:
    """Run Silero VAD without loading or invoking the Whisper model."""

    started = time.perf_counter()
    audio = decode_audio(audio_path, sampling_rate=16000)
    timestamps = get_speech_timestamps(
        audio,
        VadOptions(
            threshold=0.5,
            min_speech_duration_ms=250,
            min_silence_duration_ms=500,
            speech_pad_ms=100,
        ),
        sampling_rate=16000,
    )
    speech_seconds = sum(
        max(0, int(item["end"]) - int(item["start"])) / 16000.0
        for item in timestamps
    )
    duration_seconds = len(audio) / 16000.0
    return {
        "status": "speech" if speech_seconds >= SPEECH_GATE_MINIMUM_SEC else "no_speech",
        "gate": "silero_vad",
        "sampleRate": 16000,
        "durationSec": round(duration_seconds, 3),
        "speechSec": round(speech_seconds, 3),
        "speechRatio": round(speech_seconds / duration_seconds, 6)
        if duration_seconds > 0.0
        else 0.0,
        "minimumSpeechSec": SPEECH_GATE_MINIMUM_SEC,
        "intervals": [
            {
                "start": round(int(item["start"]) / 16000.0, 3),
                "end": round(int(item["end"]) / 16000.0, 3),
            }
            for item in timestamps
        ],
        "elapsedMs": round((time.perf_counter() - started) * 1000, 3),
    }


def normalized_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text).lower()
    return "".join(character for character in normalized if character.isalnum())


def repeated_segment_ratio(segments: list[dict[str, Any]]) -> float:
    normalized = [normalized_text(str(segment.get("text", ""))) for segment in segments]
    normalized = [text for text in normalized if text]
    if not normalized:
        return 0.0
    counts: dict[str, int] = {}
    for text in normalized:
        counts[text] = counts.get(text, 0) + 1
    repeated = sum(count for count in counts.values() if count >= 3)
    return round(repeated / len(normalized), 6)


def recognized_speech_seconds(segments: list[dict[str, Any]]) -> float:
    """Return the union of ASR segment intervals, excluding natural pauses.

    faster-whisper's duration_after_vad is not the amount of spoken language: music
    and ambient sound can keep VAD open for almost the whole video.  Comparing ASR
    segment duration with that value therefore rejects complete transcripts.  Two
    independent decodes are the reliable reference for transcript completeness.
    """
    intervals = sorted(
        (
            max(0.0, float(segment.get("start", 0.0))),
            max(0.0, float(segment.get("end", segment.get("start", 0.0)))),
        )
        for segment in segments
    )
    merged: list[list[float]] = []
    for start, end in intervals:
        end = max(start, end)
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return round(sum(end - start for start, end in merged), 3)


def build_verification(primary: dict[str, Any], second: dict[str, Any]) -> dict[str, Any]:
    primary_text = normalized_text(primary["text"])
    second_text = normalized_text(second["text"])
    agreement = difflib.SequenceMatcher(None, primary_text, second_text).ratio()
    primary_last_end = max(
        (float(segment["end"]) for segment in primary["segments"]), default=0.0
    )
    second_last_end = max(
        (float(segment["end"]) for segment in second["segments"]), default=0.0
    )
    primary_repeated_ratio = repeated_segment_ratio(primary["segments"])
    second_repeated_ratio = repeated_segment_ratio(second["segments"])
    primary_speech_sec = recognized_speech_seconds(primary["segments"])
    second_speech_sec = recognized_speech_seconds(second["segments"])
    longer_speech_sec = max(primary_speech_sec, second_speech_sec)
    speech_duration_agreement = (
        min(primary_speech_sec, second_speech_sec) / longer_speech_sec
        if longer_speech_sec > 0.0
        else 1.0
    )
    status_matches = primary["status"] == second["status"]
    content_matches = (
        primary["status"] == "no_speech"
        or (
            agreement >= VERIFY_MIN_TEXT_AGREEMENT
            and abs(primary_last_end - second_last_end)
            <= VERIFY_MAX_END_DIFFERENCE_SEC
            and primary_repeated_ratio < 0.30
            and second_repeated_ratio < 0.30
            and speech_duration_agreement >= VERIFY_MIN_SPEECH_DURATION_AGREEMENT
        )
    )
    return {
        "status": "verified" if status_matches and content_matches else "review_required",
        "strategy": "independent_changed_chunk_boundaries",
        "primaryChunkLengthSec": CHUNK_LENGTH_SEC,
        "verificationChunkLengthSec": VERIFY_CHUNK_LENGTH_SEC,
        "textAgreement": round(agreement, 6),
        "minimumTextAgreement": VERIFY_MIN_TEXT_AGREEMENT,
        "primaryTextChars": len(primary_text),
        "verificationTextChars": len(second_text),
        "primarySegmentCount": len(primary["segments"]),
        "verificationSegmentCount": len(second["segments"]),
        "primaryLastEndSec": round(primary_last_end, 3),
        "verificationLastEndSec": round(second_last_end, 3),
        "maximumEndDifferenceSec": VERIFY_MAX_END_DIFFERENCE_SEC,
        "primaryRepeatedSegmentRatio": primary_repeated_ratio,
        "verificationRepeatedSegmentRatio": second_repeated_ratio,
        "primaryRecognizedSpeechSec": primary_speech_sec,
        "verificationRecognizedSpeechSec": second_speech_sec,
        "speechDurationAgreement": round(speech_duration_agreement, 6),
        "minimumSpeechDurationAgreement": VERIFY_MIN_SPEECH_DURATION_AGREEMENT,
        "verificationTextSha256": hashlib.sha256(second_text.encode("utf-8")).hexdigest(),
    }


def transcribe(
    audio_path: str, language: str | None, priority: str = "P1"
) -> dict[str, Any]:
    started = time.perf_counter()
    with _admission_client.lease(priority), _inference_gate.acquire(priority):
        primary = decode(audio_path, language, CHUNK_LENGTH_SEC)
        second = (
            decode(audio_path, language, VERIFY_CHUNK_LENGTH_SEC)
            if VERIFY_ENABLED
            else primary
        )
    verification = build_verification(primary, second)
    return {
        "schemaVersion": 1,
        "status": primary["status"]
        if verification["status"] == "verified"
        else "review_required",
        "model": MODEL_NAME,
        "language": primary["language"],
        "languageProbability": primary["languageProbability"],
        "durationSec": primary["durationSec"],
        "durationAfterVadSec": primary["durationAfterVadSec"],
        "decoding": {
            "chunkLengthSec": CHUNK_LENGTH_SEC,
            "conditionOnPreviousText": CONDITION_ON_PREVIOUS_TEXT,
        },
        "verification": verification,
        "text": primary["text"],
        "segments": primary["segments"],
        "elapsedMs": round((time.perf_counter() - started) * 1000, 3),
    }


class Handler(BaseHTTPRequestHandler):
    server_version = f"{SERVICE_NAME}/{SERVICE_VERSION}"

    def log_message(self, fmt: str, *args: object) -> None:
        print(f"[{self.log_date_time_string()}] {fmt % args}", flush=True)

    def send_json(self, status: HTTPStatus, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        self.send_response(status.value)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        with suppress(BrokenPipeError, ConnectionResetError):
            self.wfile.write(body)
            # Worker 被停止时客户端会主动断开; 推理结果无需再回写, 也不应污染服务日志。

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
                "computeType": COMPUTE_TYPE,
                "chunkLengthSec": CHUNK_LENGTH_SEC,
                "conditionOnPreviousText": CONDITION_ON_PREVIOUS_TEXT,
                "verificationEnabled": VERIFY_ENABLED,
                "verificationChunkLengthSec": VERIFY_CHUNK_LENGTH_SEC,
                "verificationMinimumTextAgreement": VERIFY_MIN_TEXT_AGREEMENT,
                "verificationMinimumSpeechDurationAgreement": VERIFY_MIN_SPEECH_DURATION_AGREEMENT,
                "scheduler": _inference_gate.snapshot(),
            },
        )

    def do_POST(self) -> None:
        route = self.path.split("?", 1)[0]
        if route not in ("/v1/transcriptions", "/v1/speech-gate"):
            self.send_json(HTTPStatus.NOT_FOUND, {"detail": "not found"})
            return

        try:
            content_length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            content_length = 0
        if content_length <= 0:
            self.send_json(HTTPStatus.BAD_REQUEST, {"detail": "音频内容为空"})
            return
        if content_length > MAX_AUDIO_BYTES:
            self.send_json(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, {"detail": "音频文件超过大小限制"})
            return

        language_header = self.headers.get("X-ASR-Language", "auto").strip().lower()
        language = None if language_header in ("", "auto") else language_header
        temp_path = ""
        try:
            with tempfile.NamedTemporaryFile(
                prefix="gta-asr-", suffix=".flac", delete=False
            ) as temp:
                temp_path = temp.name
                remaining = content_length
                while remaining > 0:
                    chunk = self.rfile.read(min(1024 * 1024, remaining))
                    if not chunk:
                        raise ValueError("音频请求体不完整")
                    temp.write(chunk)
                    remaining -= len(chunk)
            if route == "/v1/speech-gate":
                self.send_json(HTTPStatus.OK, speech_gate(temp_path))
            else:
                priority = self.headers.get("X-GTA-Priority", "P1")
                self.send_json(
                    HTTPStatus.OK, transcribe(temp_path, language, priority)
                )
        except ValueError as error:
            self.send_json(HTTPStatus.BAD_REQUEST, {"detail": str(error)})
        except Exception as error:
            self.send_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"detail": str(error)})
        finally:
            if temp_path:
                with suppress(FileNotFoundError):
                    os.unlink(temp_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="GTA faster-whisper ASR service")
    parser.add_argument("--host", default=os.getenv("ASR_HOST", "192.168.80.7"))
    parser.add_argument("--port", type=int, default=int(os.getenv("ASR_PORT", "18084")))
    args = parser.parse_args()
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(
        f"{SERVICE_NAME} listening on {args.host}:{args.port}, model={MODEL_NAME}, "
        f"device={DEVICE}, compute_type={COMPUTE_TYPE}",
        flush=True,
    )
    server.serve_forever()


if __name__ == "__main__":
    main()
