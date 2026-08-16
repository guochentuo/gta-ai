from __future__ import annotations

import asyncio
import hashlib
import heapq
import importlib.metadata
import io
import json
import math
import os
import threading
import time
from contextlib import asynccontextmanager, contextmanager
from typing import Any

import torch
import torch.nn.functional as functional
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from PIL import Image, ImageOps, UnidentifiedImageError
from pydantic import BaseModel, Field
from transformers import AutoModel, AutoProcessor


SERVICE_NAME = "gta-ai-image-embedding"
SERVICE_VERSION = "1.1.0"
MODEL_ID = os.getenv("VISION_MODEL_ID", "google/siglip2-base-patch16-naflex")
MODEL_REVISION = os.getenv(
    "VISION_MODEL_REVISION", "b53b807d3a2d5e2b3911292f2d69e5341cdc064c"
)
MODEL_CACHE = os.getenv("VISION_MODEL_CACHE", "/data/huggingface")
CPU_THREADS = int(os.getenv("VISION_CPU_THREADS", "16"))
INTEROP_THREADS = int(os.getenv("VISION_INTEROP_THREADS", "2"))
MAX_IMAGE_BYTES = int(os.getenv("VISION_MAX_IMAGE_BYTES", str(25 * 1024 * 1024)))
MAX_IMAGE_PIXELS = int(os.getenv("VISION_MAX_IMAGE_PIXELS", "50000000"))
MAX_IMAGE_BATCH = int(os.getenv("VISION_MAX_IMAGE_BATCH", "8"))
MAX_TEXT_BATCH = int(os.getenv("VISION_MAX_TEXT_BATCH", "32"))
MASK_POLICY = "original_with_subtitle_patch_attention_mask_v1"
MAX_MASK_BOXES_PER_IMAGE = 32
MAX_EXCLUDED_PATCH_RATIO = 0.40

Image.MAX_IMAGE_PIXELS = MAX_IMAGE_PIXELS

_model: Any = None
_processor: Any = None
_dimensions = 0


def workload_priority(value: str | None) -> int:
    normalized = (value or "P1").strip().upper()
    return 0 if normalized == "P0" else 9 if normalized == "P9" else 1


class PriorityInferenceGate:
    """保护单个 SigLIP2 实例，并让等待中的实时请求越过历史补齐。"""

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


class TextEmbeddingRequest(BaseModel):
    input: str | list[str]
    model: str | None = None
    encoding_format: str = Field(default="float", pattern="^float$")


def package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


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


def normalized_vectors(features: torch.Tensor) -> list[list[float]]:
    normalized = functional.normalize(features.float(), p=2, dim=-1)
    return normalized.cpu().tolist()


def parse_subtitle_masks(raw: str | None, image_count: int) -> list[list[list[float]]]:
    if raw is None or not raw.strip():
        return [[] for _ in range(image_count)]
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as error:
        raise HTTPException(status_code=400, detail="subtitle_masks不是合法JSON") from error
    if not isinstance(payload, list) or len(payload) != image_count:
        raise HTTPException(status_code=400, detail="subtitle_masks数量必须与图片数量一致")
    normalized: list[list[list[float]]] = []
    for image_masks in payload:
        if not isinstance(image_masks, list) or len(image_masks) > MAX_MASK_BOXES_PER_IMAGE:
            raise HTTPException(status_code=400, detail="单张图片字幕框数量不符合限制")
        boxes: list[list[float]] = []
        for box in image_masks:
            if (
                not isinstance(box, list)
                or len(box) != 4
                or any(not isinstance(value, (int, float)) for value in box)
            ):
                raise HTTPException(status_code=400, detail="字幕框必须是四个归一化数值")
            left, top, right, bottom = (float(value) for value in box)
            if (
                not all(math.isfinite(value) for value in (left, top, right, bottom))
                or left < 0.0
                or top < 0.0
                or right > 1.0
                or bottom > 1.0
                or right <= left
                or bottom <= top
            ):
                raise HTTPException(status_code=400, detail="字幕框坐标范围非法")
            boxes.append([left, top, right, bottom])
        normalized.append(boxes)
    return normalized


def apply_subtitle_patch_masks(
    inputs: dict[str, torch.Tensor], masks: list[list[list[float]]]
) -> list[dict[str, Any]]:
    attention = inputs.get("pixel_attention_mask")
    spatial_shapes = inputs.get("spatial_shapes")
    if attention is None or spatial_shapes is None:
        raise RuntimeError("SigLIP2 processor没有返回patch attention mask")
    statistics: list[dict[str, Any]] = []
    for image_index, boxes in enumerate(masks):
        patch_height = int(spatial_shapes[image_index][0].item())
        patch_width = int(spatial_shapes[image_index][1].item())
        valid_patch_count = patch_height * patch_width
        before = int(attention[image_index, :valid_patch_count].sum().item())
        for left, top, right, bottom in boxes:
            left_patch = max(0, min(patch_width, math.floor(left * patch_width)))
            right_patch = max(0, min(patch_width, math.ceil(right * patch_width)))
            top_patch = max(0, min(patch_height, math.floor(top * patch_height)))
            bottom_patch = max(0, min(patch_height, math.ceil(bottom * patch_height)))
            for row in range(top_patch, bottom_patch):
                start = row * patch_width + left_patch
                end = row * patch_width + right_patch
                attention[image_index, start:end] = 0
        after = int(attention[image_index, :valid_patch_count].sum().item())
        excluded = before - after
        excluded_ratio = excluded / before if before else 0.0
        if after <= 0 or excluded_ratio > MAX_EXCLUDED_PATCH_RATIO:
            raise HTTPException(status_code=400, detail="字幕框排除的视觉patch比例过高")
        statistics.append(
            {
                "policy": MASK_POLICY,
                "box_count": len(boxes),
                "valid_patch_count": before,
                "excluded_patch_count": excluded,
                "excluded_patch_ratio": round(excluded_ratio, 6),
            }
        )
    return statistics


def embedding_input_sha256(content: bytes, boxes: list[list[float]]) -> str:
    digest = hashlib.sha256()
    digest.update(content)
    digest.update(b"\0")
    digest.update(MODEL_ID.encode("utf-8"))
    digest.update(b"\0")
    digest.update(MODEL_REVISION.encode("utf-8"))
    digest.update(b"\0")
    digest.update(MASK_POLICY.encode("utf-8"))
    digest.update(b"\0")
    digest.update(json.dumps(boxes, separators=(",", ":")).encode("utf-8"))
    return digest.hexdigest()


def embed_images(
    images: list[Image.Image], masks: list[list[list[float]]],
    priority_header: str | None = "P1",
) -> tuple[list[list[float]], list[dict[str, Any]]]:
    if _model is None or _processor is None:
        raise RuntimeError("图片向量模型尚未加载")
    with _inference_gate.hold(priority_header), torch.inference_mode():
        inputs = _processor(images=images, return_tensors="pt")
        mask_statistics = apply_subtitle_patch_masks(inputs, masks)
        features = _model.get_image_features(**inputs)
        return normalized_vectors(features), mask_statistics


def embed_texts(
    texts: list[str], priority_header: str | None = "P1"
) -> list[list[float]]:
    if _model is None or _processor is None:
        raise RuntimeError("图片向量模型尚未加载")
    with _inference_gate.hold(priority_header), torch.inference_mode():
        inputs = _processor(
            text=texts,
            padding="max_length",
            max_length=64,
            truncation=True,
            return_tensors="pt",
        )
        features = _model.get_text_features(**inputs)
        return normalized_vectors(features)


@asynccontextmanager
async def lifespan(_: FastAPI):
    global _model, _processor, _dimensions
    torch.set_num_threads(CPU_THREADS)
    torch.set_num_interop_threads(INTEROP_THREADS)
    _processor = AutoProcessor.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        cache_dir=MODEL_CACHE,
    )
    _model = AutoModel.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        cache_dir=MODEL_CACHE,
        torch_dtype=torch.float32,
    ).eval()
    probe = embed_texts(["health probe"])
    _dimensions = len(probe[0])
    yield
    _model = None
    _processor = None
    _dimensions = 0


app = FastAPI(title=SERVICE_NAME, version=SERVICE_VERSION, lifespan=lifespan)


@app.get("/health")
def health() -> dict[str, Any]:
    if _model is None or _processor is None or _dimensions <= 0:
        raise HTTPException(status_code=503, detail="图片向量模型尚未加载")
    return {
        "status": "ok",
        "service": SERVICE_NAME,
        "version": SERVICE_VERSION,
        "model": MODEL_ID,
        "revision": MODEL_REVISION,
        "dimensions": _dimensions,
        "normalized": True,
        "maskPolicy": MASK_POLICY,
        "runtime": {
            "torch": package_version("torch"),
            "transformers": package_version("transformers"),
        },
        "scheduler": _inference_gate.snapshot(),
    }


@app.post("/v1/image-embeddings")
async def image_embeddings(
    request: Request,
    files: list[UploadFile] = File(...),
    subtitle_masks: str | None = Form(default=None),
) -> dict[str, Any]:
    if not files or len(files) > MAX_IMAGE_BATCH:
        raise HTTPException(status_code=400, detail="图片批次大小不符合限制")

    contents = [await file.read(MAX_IMAGE_BYTES + 1) for file in files]
    images = [decode_image(content) for content in contents]
    masks = parse_subtitle_masks(subtitle_masks, len(images))
    started = time.perf_counter()
    vectors, mask_statistics = await asyncio.to_thread(
        embed_images, images, masks, request.headers.get("X-GTA-Priority")
    )
    elapsed_ms = round((time.perf_counter() - started) * 1000, 3)
    return {
        "object": "list",
        "model": MODEL_ID,
        "revision": MODEL_REVISION,
        "dimensions": _dimensions,
        "normalized": True,
        "input_policy": MASK_POLICY,
        "data": [
            {
                "object": "embedding",
                "index": index,
                "embedding": vector,
                "source_sha256": hashlib.sha256(contents[index]).hexdigest(),
                "embedding_input_sha256": embedding_input_sha256(contents[index], masks[index]),
                "width": images[index].width,
                "height": images[index].height,
                "subtitle_patch_mask": mask_statistics[index],
            }
            for index, vector in enumerate(vectors)
        ],
        "elapsed_ms": elapsed_ms,
    }


@app.post("/v1/text-embeddings")
async def text_embeddings(
    payload: TextEmbeddingRequest, request: Request
) -> dict[str, Any]:
    texts = [payload.input] if isinstance(payload.input, str) else payload.input
    texts = [text.strip() for text in texts]
    if not texts or len(texts) > MAX_TEXT_BATCH or any(not text for text in texts):
        raise HTTPException(status_code=400, detail="文本批次大小或内容不符合限制")

    started = time.perf_counter()
    vectors = await asyncio.to_thread(
        embed_texts, texts, request.headers.get("X-GTA-Priority")
    )
    elapsed_ms = round((time.perf_counter() - started) * 1000, 3)
    return {
        "object": "list",
        "model": MODEL_ID,
        "revision": MODEL_REVISION,
        "dimensions": _dimensions,
        "normalized": True,
        "data": [
            {"object": "embedding", "index": index, "embedding": vector}
            for index, vector in enumerate(vectors)
        ],
        "elapsed_ms": elapsed_ms,
    }
