from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import Field, field_validator

from gta_ai.schemas.base import StrictModel


class MediaAuditRequest(StrictModel):
    """Java 提交给 gta-ai 的无数据库审核契约。"""

    schema_version: Literal[1] = 1
    request_id: str = Field(min_length=8, max_length=64)
    asset_id: str = Field(min_length=1, max_length=64)
    asset_generation: str = Field(min_length=8, max_length=64)
    media_type: Literal["image", "video"]
    source_key: str = Field(min_length=1, max_length=2048)
    source_md5: str = Field(min_length=16, max_length=128)
    audit_input_hash: str = Field(min_length=32, max_length=128)
    policy_version: str = Field(min_length=1, max_length=64)
    priority: Literal["P1", "P9"] = "P1"
    requested_at: datetime

    @field_validator("requested_at", mode="before")
    @classmethod
    def parse_wire_datetime(cls, value: object) -> object:
        if isinstance(value, str):
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        return value


class MediaAuditAccepted(StrictModel):
    request_id: str
    state: Literal["queued", "running", "completed"]
    idempotent: bool = False


class MediaAuditState(StrictModel):
    request_id: str
    state: Literal["queued", "running", "completed", "failed", "cancelled", "unknown"]
    detail: str | None = None
