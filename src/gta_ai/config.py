from __future__ import annotations

from functools import lru_cache
from typing import Literal, Self

from pydantic import AnyHttpUrl, Field, SecretStr, field_validator, model_validator
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

    local_llm_base_url: AnyHttpUrl = "http://192.168.80.7:8000/v1"
    local_llm_model: str = "Qwen/Qwen3.6-27B-FP8"
    local_llm_api_key: SecretStr | None = None
    local_llm_timeout_seconds: float = Field(default=600, gt=0, le=3600)
    local_llm_health_timeout_seconds: float = Field(default=3, gt=0, le=30)
    internal_inference_api_token: SecretStr | None = None

    openai_api_key: SecretStr | None = None
    openai_model: str = "gpt-5.6-sol"
    openai_reasoning_effort: ReasoningEffort = "high"
    openai_timeout_seconds: float = Field(default=300, gt=0, le=3600)

    # 素材审核编排只使用 ES/Ceph/模型。gta-ai 不得配置、连接或写入业务数据库。
    media_audit_enabled: bool = False
    media_audit_api_token: SecretStr | None = None
    media_audit_elasticsearch_url: AnyHttpUrl = "http://127.0.0.1:9200"
    media_audit_elasticsearch_username: str = ""
    media_audit_elasticsearch_password: SecretStr | None = None
    media_audit_elasticsearch_verify_tls: bool = True
    media_audit_material_alias: str = "gta_media_admin"
    media_audit_result_alias: str = "gta_media_content_audit"
    media_audit_java_callback_url: AnyHttpUrl | None = None
    media_audit_java_callback_secret: SecretStr | None = None
    media_audit_max_concurrency: int = Field(default=1, ge=1, le=3)
    media_audit_request_timeout_seconds: float = Field(default=900, gt=0, le=3600)
    media_audit_callback_timeout_seconds: float = Field(default=15, gt=0, le=120)
    media_audit_projection_retry_seconds: float = Field(default=5, gt=0, le=300)
    # 技术失败由 Java 在全部历史首轮完成后统一重试; AI 内部不能热循环抢占 P1/P9.
    media_audit_transient_retry_max_attempts: int = Field(default=0, ge=0, le=10)
    media_audit_transient_retry_seconds: float = Field(default=5, gt=0, le=300)
    media_audit_log_path: str = ""
    # 27B 当前实测约 7 token/s。结构化审核 JSON 不应超过 HTTP 10 分钟
    # 窗口; 2200 token 足够容纳完整问题列表, 且避免 5000 token 稳定超时。
    media_audit_max_output_tokens: int = Field(default=2200, ge=512, le=4000)
    # 图片视觉理解已经由 worker 生成. 终审只读取结构化摘要并返回短 JSON,
    # 不允许再次走视频级长输出路径.
    media_audit_image_max_output_tokens: int = Field(default=700, ge=256, le=1600)
    media_audit_kafka_bootstrap_servers: str = ""
    media_audit_kafka_request_topic: str = "gta.media.audit.request.v1"
    media_audit_kafka_cancel_topic: str = "gta.media.audit.cancel.v1"
    media_audit_kafka_result_topic: str = "gta.media.audit.result.v1"
    media_audit_kafka_group_id: str = "gta-ai-media-audit-v1"
    media_audit_kafka_auto_offset_reset: Literal["earliest", "latest"] = "earliest"
    media_audit_kafka_poll_timeout_seconds: float = Field(default=0.5, gt=0, le=10)

    # gta-ai owns the read-only ES adapter; Java never receives ES configuration.
    material_search_enabled: bool = True
    material_search_timeout_seconds: float = Field(default=15, gt=0, le=120)
    material_search_embedding_enabled: bool = True
    material_search_embedding_url: AnyHttpUrl = "http://192.168.80.130:18080/v1/embeddings"
    material_search_embedding_model: str = "Qwen/Qwen3-Embedding-0.6B"
    material_search_dimensions: int = Field(default=1024, ge=1, le=8192)

    @field_validator(
        "local_llm_api_key",
        "internal_inference_api_token",
        "openai_api_key",
        "media_audit_api_token",
        "media_audit_elasticsearch_password",
        "media_audit_java_callback_secret",
        mode="before",
    )
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

    @model_validator(mode="after")
    def validate_media_audit_boundary(self) -> Self:
        if not self.media_audit_enabled:
            return self
        missing: list[str] = []
        if not self.media_audit_kafka_bootstrap_servers.strip():
            missing.append("GTA_AI_MEDIA_AUDIT_KAFKA_BOOTSTRAP_SERVERS")
        if self.media_audit_java_callback_url is None:
            missing.append("GTA_AI_MEDIA_AUDIT_JAVA_CALLBACK_URL")
        if self.media_audit_java_callback_secret is None:
            missing.append("GTA_AI_MEDIA_AUDIT_JAVA_CALLBACK_SECRET")
        if missing:
            raise ValueError("media audit enabled but missing: " + ", ".join(missing))
        return self

    @classmethod
    def for_tests(cls, **overrides: object) -> Self:
        if overrides.get("media_audit_enabled"):
            overrides.setdefault("media_audit_kafka_bootstrap_servers", "test-kafka:9092")
        return cls(_env_file=None, app_env="test", **overrides)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
