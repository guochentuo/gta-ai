from __future__ import annotations

from typing import Literal

from pydantic import Field

from gta_ai.schemas.base import StrictModel


class ModelMessage(StrictModel):
    role: Literal["developer", "system", "user", "assistant"]
    content: str = Field(min_length=1)


class LocalGenerationRequest(StrictModel):
    messages: list[ModelMessage] = Field(min_length=1)
    max_tokens: int = Field(default=2048, ge=1, le=32768)
    temperature: float = Field(default=0.2, ge=0, le=2)


class LocalGenerationResponse(StrictModel):
    provider: Literal["local"] = "local"
    model: str = Field(min_length=1)
    content: str
    finish_reason: str | None = None
