from __future__ import annotations

from typing import Any, Literal

from pydantic import Field

from gta_ai.schemas.base import StrictModel


class MaterialSearchRequest(StrictModel):
    keyword: str | None = Field(default=None, max_length=1000)
    type: str | None = Field(default=None, max_length=16)
    user_id: int | None = None
    biz_id: str | None = Field(default=None, max_length=64)
    status: str | None = Field(default=None, max_length=32)
    orientation: Literal["landscape", "portrait", "square"] | None = None
    use_type: str | None = Field(default=None, max_length=256)
    start_date: str | None = Field(default=None, max_length=32)
    end_date: str | None = Field(default=None, max_length=32)
    remark: str | None = Field(default=None, max_length=1000)
    original_author: str | None = Field(default=None, max_length=256)
    is_reject: str | None = Field(default=None, max_length=8)
    min_score: int | None = Field(default=None, ge=0, le=100)
    max_score: int | None = Field(default=None, ge=0, le=100)
    page_num: int = Field(default=1, ge=1, le=100_000)
    page_size: int = Field(default=10, ge=1, le=100)


class MaterialSearchResult(StrictModel):
    total: int = Field(ge=0)
    list: list[dict[str, Any]]
