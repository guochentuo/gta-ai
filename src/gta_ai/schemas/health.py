from __future__ import annotations

from enum import StrEnum

from pydantic import Field

from gta_ai.schemas.base import StrictModel


class HealthState(StrEnum):
    OK = "ok"
    CONFIGURED = "configured"
    DEGRADED = "degraded"
    UNAVAILABLE = "unavailable"
    NOT_CONFIGURED = "not_configured"


class ProviderHealth(StrictModel):
    provider: str = Field(min_length=1)
    state: HealthState
    model: str | None = None
    latency_ms: float | None = Field(default=None, ge=0)
    detail: str | None = None


class ServiceHealth(StrictModel):
    service: str = "gta-ai"
    version: str
    state: HealthState
    providers: list[ProviderHealth] = Field(default_factory=list)
