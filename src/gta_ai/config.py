from __future__ import annotations

from functools import lru_cache
from typing import Literal, Self

from pydantic import AnyHttpUrl, Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

ReasoningEffort = Literal["none", "low", "medium", "high", "xhigh", "max"]
Environment = Literal["development", "test", "production"]


class Settings(BaseSettings):
    """Runtime configuration loaded exclusively from GTA_AI_* environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="GTA_AI_",
        case_sensitive=False,
        extra="ignore",
    )

    app_name: str = "gta-ai"
    app_env: Environment = "development"
    host: str = "127.0.0.1"
    port: int = Field(default=8080, ge=1, le=65535)
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"

    local_llm_base_url: AnyHttpUrl = "http://127.0.0.1:8000/v1"
    local_llm_model: str = "Qwen/Qwen3.6-27B-FP8"
    local_llm_api_key: SecretStr | None = None
    local_llm_timeout_seconds: float = Field(default=600, gt=0, le=3600)
    local_llm_health_timeout_seconds: float = Field(default=3, gt=0, le=30)

    openai_api_key: SecretStr | None = None
    openai_model: str = "gpt-5.6-sol"
    openai_reasoning_effort: ReasoningEffort = "high"
    openai_timeout_seconds: float = Field(default=300, gt=0, le=3600)

    @field_validator("local_llm_api_key", "openai_api_key", mode="before")
    @classmethod
    def empty_secret_is_none(cls, value: object) -> object:
        if isinstance(value, str) and not value.strip():
            return None
        return value

    @field_validator("app_name", "local_llm_model", "openai_model")
    @classmethod
    def non_empty_text(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("must not be empty")
        return value

    def local_llm_url(self, path: str) -> str:
        base = str(self.local_llm_base_url).rstrip("/")
        return f"{base}/{path.lstrip('/')}"

    @property
    def openai_is_configured(self) -> bool:
        return self.openai_api_key is not None

    @classmethod
    def for_tests(cls, **overrides: object) -> Self:
        return cls(_env_file=None, app_env="test", **overrides)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
