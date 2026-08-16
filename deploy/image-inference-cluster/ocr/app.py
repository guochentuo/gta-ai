from __future__ import annotations

import asyncio
import hashlib
import heapq
import importlib.metadata
import io
import os
import threading
import time
from contextlib import asynccontextmanager, contextmanager
from typing import Any

import numpy as np
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from paddleocr import PaddleOCR
from PIL import Image, ImageOps, UnidentifiedImageError


SERVICE_NAME = "gta-ai-ocr"
SERVICE_VERSION = "1.1.0"
DETECTION_MODEL = os.getenv("OCR_DETECTION_MODEL", "PP-OCRv6_medium_det")
RECOGNITION_MODEL = os.getenv("OCR_RECOGNITION_MODEL", "PP-OCRv6_medium_rec")
CPU_THREADS = int(os.getenv("OCR_CPU_THREADS", "8"))
MAX_IMAGE_BYTES = int(os.getenv("OCR_MAX_IMAGE_BYTES", str(25 * 1024 * 1024)))
MAX_IMAGE_PIXELS = int(os.getenv("OCR_MAX_IMAGE_PIXELS", "50000000"))
MIN_SCORE = float(os.getenv("OCR_MIN_SCORE", "0.0"))
MAX_BATCH_SIZE = int(os.getenv("OCR_MAX_BATCH_SIZE", "32"))

Image.MAX_IMAGE_PIXELS = MAX_IMAGE_PIXELS

_engine: PaddleOCR | None = None


def workload_priority(value: str | None) -> int:
    normalized = (value or "P1").strip().upper()
    return 0 if normalized == "P0" else 9 if normalized == "P9" else 1


class PriorityInferenceGate:
    """单模型串行执行；等待中的实时任务永远先于历史补齐。"""

    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._waiting: list[tuple[int, int, object]] = []
        self._sequence = 0
        self._active = False

    @contextmanager
    def hold(self, priority_header: str | None):
        ticket = object()
        with self._condition:
            self._sequence += 1
            heapq.heappush(
                self._waiting,
                (workload_priority(priority_header), self._sequence, ticket),
            )
            self._condition.wait_for(
                lambda: not self._active and self._waiting[0][2] is ticket
            )
            heapq.heappop(self._waiting)
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
                "waiting_realtime": sum(item[0] <= 1 for item in self._waiting),
                "waiting_historical": sum(item[0] == 9 for item in self._waiting),
            }


_inference_gate = PriorityInferenceGate()


def package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def json_compatible(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): json_compatible(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_compatible(item) for item in value]
    return value


def result_payload(result: Any) -> dict[str, Any]:
    raw = getattr(result, "json", result)
    if callable(raw):
        raw = raw()
    raw = json_compatible(raw)
    if not isinstance(raw, dict):
        return {}
    nested = raw.get("res")
    return nested if isinstance(nested, dict) else raw


def decode_image(content: bytes) -> Image.Image:
    if not content:
        raise HTTPException(status_code=400, detail="图片内容为空")
    if len(content) > MAX_IMAGE_BYTES:
        raise HTTPException(status_code=413, detail="图片文件超过大小限制")

    try:
        image = Image.open(io.BytesIO(content))
        width, height = image.size
        if width <= 0 or height <= 0 or width * height > MAX_IMAGE_PIXELS:
            raise HTTPException(status_code=413, detail="图片像素数量超过限制")
        image = ImageOps.exif_transpose(image)
        return image.convert("RGB")
    except HTTPException:
        raise
    except (Image.DecompressionBombError, UnidentifiedImageError, OSError) as error:
        raise HTTPException(status_code=415, detail="无法解码图片") from error


def lines_from_result(result: Any) -> list[dict[str, Any]]:
    lines: list[dict[str, Any]] = []
    payload = result_payload(result)
    texts = payload.get("rec_texts") or []
    scores = payload.get("rec_scores") or []
    polygons = payload.get("rec_polys") or payload.get("dt_polys") or []
    boxes = payload.get("rec_boxes") or []

    for index, text in enumerate(texts):
        normalized_text = str(text).strip()
        score = float(scores[index]) if index < len(scores) else 0.0
        if not normalized_text or score < MIN_SCORE:
            continue
        polygon = polygons[index] if index < len(polygons) else None
        box = boxes[index] if index < len(boxes) else None
        lines.append(
            {
                "text": normalized_text,
                "score": score,
                "polygon": json_compatible(polygon),
                "box": json_compatible(box),
            }
        )
    return lines


def run_ocr_batch(
    images: list[Image.Image], priority_header: str | None = "P1"
) -> list[list[dict[str, Any]]]:
    if _engine is None:
        raise RuntimeError("OCR模型尚未加载")
    if not images:
        return []

    # PaddleOCR 原生接受图片列表。一个批次只进入一次模型调用，避免逐图 HTTP、
    # 预处理和调度开销；进程内仍由锁保护同一推理引擎。
    with _inference_gate.hold(priority_header):
        results = list(_engine.predict([np.asarray(image) for image in images]))
    if len(results) != len(images):
        raise RuntimeError(
            f"OCR批量结果数量不匹配: expected={len(images)}, actual={len(results)}"
        )
    return [lines_from_result(result) for result in results]


def ocr_payload(content: bytes, image: Image.Image, lines: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "model": f"{DETECTION_MODEL}+{RECOGNITION_MODEL}",
        "source_sha256": hashlib.sha256(content).hexdigest(),
        "width": image.width,
        "height": image.height,
        "text": "\n".join(line["text"] for line in lines),
        "lines": lines,
    }


@asynccontextmanager
async def lifespan(_: FastAPI):
    global _engine
    _engine = PaddleOCR(
        device="cpu",
        text_detection_model_name=DETECTION_MODEL,
        text_recognition_model_name=RECOGNITION_MODEL,
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        enable_mkldnn=True,
        cpu_threads=CPU_THREADS,
    )
    yield
    _engine = None


app = FastAPI(title=SERVICE_NAME, version=SERVICE_VERSION, lifespan=lifespan)


@app.get("/health")
def health() -> dict[str, Any]:
    if _engine is None:
        raise HTTPException(status_code=503, detail="OCR模型尚未加载")
    return {
        "status": "ok",
        "service": SERVICE_NAME,
        "version": SERVICE_VERSION,
        "models": {
            "detection": DETECTION_MODEL,
            "recognition": RECOGNITION_MODEL,
        },
        "runtime": {
            "paddlepaddle": package_version("paddlepaddle"),
            "paddleocr": package_version("paddleocr"),
        },
        "limits": {"max_batch_size": MAX_BATCH_SIZE},
        "scheduler": _inference_gate.snapshot(),
    }


@app.post("/v1/ocr")
async def ocr(request: Request, file: UploadFile = File(...)) -> dict[str, Any]:
    content = await file.read(MAX_IMAGE_BYTES + 1)
    image = decode_image(content)
    started = time.perf_counter()
    lines = (
        await asyncio.to_thread(
            run_ocr_batch, [image], request.headers.get("X-GTA-Priority")
        )
    )[0]
    elapsed_ms = round((time.perf_counter() - started) * 1000, 3)
    return {**ocr_payload(content, image, lines), "elapsed_ms": elapsed_ms}


@app.post("/v1/ocr-batch")
async def ocr_batch(
    request: Request, files: list[UploadFile] = File(...)
) -> dict[str, Any]:
    if not files or len(files) > MAX_BATCH_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"OCR批量图片数量必须在1到{MAX_BATCH_SIZE}之间",
        )
    contents = [await file.read(MAX_IMAGE_BYTES + 1) for file in files]
    images = [decode_image(content) for content in contents]
    started = time.perf_counter()
    batch_lines = await asyncio.to_thread(
        run_ocr_batch, images, request.headers.get("X-GTA-Priority")
    )
    elapsed_ms = round((time.perf_counter() - started) * 1000, 3)
    items = [
        {
            "index": index,
            **ocr_payload(contents[index], images[index], batch_lines[index]),
        }
        for index in range(len(images))
    ]
    return {
        "model": f"{DETECTION_MODEL}+{RECOGNITION_MODEL}",
        "count": len(items),
        "items": items,
        "elapsed_ms": elapsed_ms,
    }
