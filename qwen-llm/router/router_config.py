# ruff: noqa: RUF002
"""27B Router运行配置。

只负责把环境配置转换为强类型对象，不承载接口或业务流程。
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from .retrieval import RetrievalConfig


@dataclass(frozen=True)
class RouterConfig:
    host: str = "127.0.0.1"
    port: int = 7100
    backend_url: str = "http://127.0.0.1:7106"
    internal_model: str = "Qwen/Qwen3.8-27B-FP8"
    state_path: Path = Path("/opt/gta-ai/qwen-llm/state/router-state.json")
    gpu_lock_path: Path = Path("/opt/gta-ai/qwen-llm/state/gpu.lock")
    wake_timeout_seconds: float = 300.0
    backend_poll_seconds: float = 0.25
    request_timeout_seconds: float = 1800.0
    max_concurrent_workloads: int = 1
    admission_db_path: Path = Path("/opt/gta-ai/qwen-llm/state/gpu-admission.sqlite3")
    admission_poll_seconds: float = 0.1
    admission_minimum_free_mb: int = 2048
    admission_max_active: int = 1
    startup_benchmark_enabled: bool = True
    startup_benchmark_max_tokens: int = 32
    standard_max_tokens: int = 2048
    persona_prompt_enabled: bool = True
    max_history_messages: int = 8
    max_history_chars: int = 4000
    retrieval_enabled: bool = False
    retrieval_embedding_url: str = "http://192.168.80.7:7101/v1/embeddings"
    retrieval_embedding_model: str = "Qwen/Qwen3-Embedding-0.6B"
    retrieval_elasticsearch_url: str = "https://192.168.80.130:9200"
    retrieval_elasticsearch_username: str = ""
    retrieval_elasticsearch_password: str = ""
    retrieval_elasticsearch_indices: tuple[str, ...] = RetrievalConfig.elasticsearch_indices
    retrieval_verify_tls: bool = False
    retrieval_timeout_seconds: float = 1.5
    retrieval_top_k: int = 12
    retrieval_context_top_k: int = 3
    retrieval_num_candidates: int = 100
    retrieval_max_query_chars: int = 1000
    retrieval_max_context_chars: int = 2400
    retrieval_max_document_chars: int = 600
    retrieval_min_score: float = 0.72
    retrieval_image_min_score: float = 0.78
    welcome_template_enabled: bool = True
    welcome_template_index: str = "gta_ai_welcome_template_v1"
    welcome_template_timeout_seconds: float = 2.0
    welcome_template_character_interval_ms: int = 12
    welcome_localization_enabled: bool = True
    welcome_localization_url: str = "http://127.0.0.1:18086/v1/chat/completions"
    welcome_localization_model: str = "Qwen/Qwen3.8-27B-FP8"
    welcome_localization_timeout_seconds: float = 120.0
    welcome_localization_max_tokens: int = 4096
    welcome_localization_cache_key_prefix: str = "gta:ai:welcome-localized"
    welcome_localization_cache_ttl_seconds: int = 2_592_000
    welcome_localization_prewarm_locales: tuple[str, ...] = (
        "en-US",
        "zh-TW",
        "ja-JP",
        "ko-KR",
        "de-DE",
        "fr-FR",
        "es-ES",
        "th-TH",
    )
    welcome_localization_prewarm_keyword: str = "中国旅行"
    quote_url: str = "http://192.168.100.5:8087/spuFeign/quoteSpu"
    quote_timeout_seconds: float = 1.5
    summarization_enabled: bool = False
    summarization_url: str = "http://127.0.0.1:7107/v1/chat/completions"
    summarization_model: str = "Qwen3-4B-Instruct-2507-Q4_K_M"
    summarization_timeout_seconds: float = 120.0
    summarization_max_summary_chars: int = 800
    summarization_max_context_tokens: int = 1200
    summarization_recent_turns: int = 2
    redis_key_prefix: str = "gta:ai:session"
    redis_ttl_seconds: int = 2_592_000
    redis_socket_timeout_seconds: float = 2.0
    redis_cluster_nodes: tuple[str, ...] = (
        "192.168.80.4:7001",
        "192.168.80.5:7001",
        "192.168.80.6:7001",
    )
    redis_password: str = ""
    simple_chat_enabled: bool = False
    simple_chat_url: str = "http://127.0.0.1:7107"
    simple_chat_model: str = "Qwen3-4B-Instruct-2507-Q4_K_M"
    simple_chat_max_tokens: int = 192

    @classmethod
    def from_environment(cls) -> RouterConfig:
        return cls(
            host=os.getenv("GTA_AI_ROUTER_HOST", cls.host),
            port=int(os.getenv("GTA_AI_ROUTER_PORT", str(cls.port))),
            backend_url=os.getenv("GTA_AI_ROUTER_BACKEND_URL", cls.backend_url).rstrip("/"),
            internal_model=os.getenv("GTA_AI_ROUTER_INTERNAL_MODEL", cls.internal_model),
            state_path=Path(os.getenv("GTA_AI_ROUTER_STATE_PATH", str(cls.state_path))),
            gpu_lock_path=Path(os.getenv("GTA_AI_ROUTER_GPU_LOCK_PATH", str(cls.gpu_lock_path))),
            wake_timeout_seconds=float(
                os.getenv("GTA_AI_ROUTER_WAKE_TIMEOUT_SECONDS", str(cls.wake_timeout_seconds))
            ),
            backend_poll_seconds=float(
                os.getenv("GTA_AI_ROUTER_BACKEND_POLL_SECONDS", str(cls.backend_poll_seconds))
            ),
            request_timeout_seconds=float(
                os.getenv(
                    "GTA_AI_ROUTER_REQUEST_TIMEOUT_SECONDS",
                    str(cls.request_timeout_seconds),
                )
            ),
            max_concurrent_workloads=max(
                1,
                int(
                    os.getenv(
                        "GTA_AI_ROUTER_MAX_CONCURRENT_WORKLOADS",
                        str(cls.max_concurrent_workloads),
                    )
                ),
            ),
            admission_db_path=Path(
                os.getenv("GTA_AI_ADMISSION_DB_PATH", str(cls.admission_db_path))
            ),
            admission_poll_seconds=max(
                0.02,
                float(
                    os.getenv(
                        "GTA_AI_ADMISSION_POLL_SECONDS",
                        str(cls.admission_poll_seconds),
                    )
                ),
            ),
            admission_minimum_free_mb=max(
                0,
                int(
                    os.getenv(
                        "GTA_AI_ADMISSION_MINIMUM_FREE_MB",
                        str(cls.admission_minimum_free_mb),
                    )
                ),
            ),
            admission_max_active=max(
                1,
                int(
                    os.getenv(
                        "GTA_AI_ADMISSION_MAX_ACTIVE",
                        str(cls.admission_max_active),
                    )
                ),
            ),
            startup_benchmark_enabled=os.getenv(
                "GTA_AI_ROUTER_STARTUP_BENCHMARK_ENABLED", "true"
            ).lower()
            in {"1", "true", "yes"},
            startup_benchmark_max_tokens=max(
                8,
                min(
                    128,
                    int(os.getenv("GTA_AI_ROUTER_STARTUP_BENCHMARK_MAX_TOKENS", "32")),
                ),
            ),
            standard_max_tokens=max(
                1,
                int(
                    os.getenv(
                        "GTA_AI_ROUTER_STANDARD_MAX_TOKENS",
                        str(cls.standard_max_tokens),
                    )
                ),
            ),
            persona_prompt_enabled=os.getenv("GTA_AI_PERSONA_PROMPT_ENABLED", "true").lower()
            in {"1", "true", "yes"},
            max_history_messages=max(1, int(os.getenv("GTA_AI_ROUTER_MAX_HISTORY_MESSAGES", "8"))),
            max_history_chars=max(256, int(os.getenv("GTA_AI_ROUTER_MAX_HISTORY_CHARS", "4000"))),
            retrieval_enabled=os.getenv("GTA_AI_RETRIEVAL_ENABLED", "false").lower()
            in {"1", "true", "yes"},
            retrieval_embedding_url=os.getenv(
                "GTA_AI_RETRIEVAL_EMBEDDING_URL", cls.retrieval_embedding_url
            ),
            retrieval_embedding_model=os.getenv(
                "GTA_AI_RETRIEVAL_EMBEDDING_MODEL", cls.retrieval_embedding_model
            ),
            retrieval_elasticsearch_url=os.getenv(
                "GTA_AI_RETRIEVAL_ELASTICSEARCH_URL", cls.retrieval_elasticsearch_url
            ).rstrip("/"),
            retrieval_elasticsearch_username=os.getenv(
                "GTA_AI_RETRIEVAL_ELASTICSEARCH_USERNAME", ""
            ),
            retrieval_elasticsearch_password=os.getenv(
                "GTA_AI_RETRIEVAL_ELASTICSEARCH_PASSWORD", ""
            ),
            retrieval_elasticsearch_indices=tuple(
                value.strip()
                for value in os.getenv(
                    "GTA_AI_RETRIEVAL_ELASTICSEARCH_INDICES",
                    ",".join(cls.retrieval_elasticsearch_indices),
                ).split(",")
                if value.strip()
            ),
            retrieval_verify_tls=os.getenv("GTA_AI_RETRIEVAL_VERIFY_TLS", "false").lower()
            in {"1", "true", "yes"},
            retrieval_timeout_seconds=max(
                0.1, float(os.getenv("GTA_AI_RETRIEVAL_TIMEOUT_SECONDS", "1.5"))
            ),
            retrieval_top_k=max(3, int(os.getenv("GTA_AI_RETRIEVAL_TOP_K", "12"))),
            retrieval_context_top_k=max(1, int(os.getenv("GTA_AI_RETRIEVAL_CONTEXT_TOP_K", "3"))),
            retrieval_num_candidates=max(
                1, int(os.getenv("GTA_AI_RETRIEVAL_NUM_CANDIDATES", "100"))
            ),
            retrieval_max_query_chars=max(
                1, int(os.getenv("GTA_AI_RETRIEVAL_MAX_QUERY_CHARS", "1000"))
            ),
            retrieval_max_context_chars=max(
                1, int(os.getenv("GTA_AI_RETRIEVAL_MAX_CONTEXT_CHARS", "2400"))
            ),
            retrieval_max_document_chars=max(
                1, int(os.getenv("GTA_AI_RETRIEVAL_MAX_DOCUMENT_CHARS", "600"))
            ),
            retrieval_min_score=max(
                0.0, min(1.0, float(os.getenv("GTA_AI_RETRIEVAL_MIN_SCORE", "0.72")))
            ),
            retrieval_image_min_score=max(
                0.0,
                min(1.0, float(os.getenv("GTA_AI_RETRIEVAL_IMAGE_MIN_SCORE", "0.78"))),
            ),
            welcome_template_enabled=os.getenv(
                "GTA_AI_WELCOME_TEMPLATE_ENABLED", "true"
            ).lower()
            in {"1", "true", "yes"},
            welcome_template_index=os.getenv(
                "GTA_AI_WELCOME_TEMPLATE_INDEX", cls.welcome_template_index
            ),
            welcome_template_timeout_seconds=max(
                0.1,
                float(
                    os.getenv(
                        "GTA_AI_WELCOME_TEMPLATE_TIMEOUT_SECONDS",
                        str(cls.welcome_template_timeout_seconds),
                    )
                ),
            ),
            welcome_template_character_interval_ms=max(
                0,
                min(
                    100,
                    int(
                        os.getenv(
                            "GTA_AI_WELCOME_TEMPLATE_CHARACTER_INTERVAL_MS",
                            str(cls.welcome_template_character_interval_ms),
                        )
                    ),
                ),
            ),
            welcome_localization_enabled=os.getenv(
                "GTA_AI_WELCOME_LOCALIZATION_ENABLED", "true"
            ).lower()
            in {"1", "true", "yes"},
            welcome_localization_url=os.getenv(
                "GTA_AI_WELCOME_LOCALIZATION_URL", cls.welcome_localization_url
            ),
            welcome_localization_model=os.getenv(
                "GTA_AI_WELCOME_LOCALIZATION_MODEL", cls.welcome_localization_model
            ),
            welcome_localization_timeout_seconds=max(
                1.0,
                float(os.getenv("GTA_AI_WELCOME_LOCALIZATION_TIMEOUT_SECONDS", "120")),
            ),
            welcome_localization_max_tokens=max(
                256, int(os.getenv("GTA_AI_WELCOME_LOCALIZATION_MAX_TOKENS", "4096"))
            ),
            welcome_localization_cache_key_prefix=os.getenv(
                "GTA_AI_WELCOME_LOCALIZATION_CACHE_KEY_PREFIX",
                cls.welcome_localization_cache_key_prefix,
            ).strip(),
            welcome_localization_cache_ttl_seconds=max(
                60,
                int(os.getenv("GTA_AI_WELCOME_LOCALIZATION_CACHE_TTL_SECONDS", "2592000")),
            ),
            welcome_localization_prewarm_locales=tuple(
                value.strip()
                for value in os.getenv(
                    "GTA_AI_WELCOME_LOCALIZATION_PREWARM_LOCALES",
                    ",".join(cls.welcome_localization_prewarm_locales),
                ).split(",")
                if value.strip()
            ),
            welcome_localization_prewarm_keyword=os.getenv(
                "GTA_AI_WELCOME_LOCALIZATION_PREWARM_KEYWORD",
                cls.welcome_localization_prewarm_keyword,
            ).strip(),
            quote_url=os.getenv("GTA_AI_QUOTE_URL", cls.quote_url),
            quote_timeout_seconds=max(
                0.1, float(os.getenv("GTA_AI_QUOTE_TIMEOUT_SECONDS", "1.5"))
            ),
            summarization_enabled=os.getenv("GTA_AI_SUMMARIZATION_ENABLED", "false").lower()
            in {"1", "true", "yes"},
            summarization_url=os.getenv("GTA_AI_SUMMARIZATION_URL", cls.summarization_url),
            summarization_model=os.getenv("GTA_AI_SUMMARIZATION_MODEL", cls.summarization_model),
            summarization_timeout_seconds=max(
                1.0, float(os.getenv("GTA_AI_SUMMARIZATION_TIMEOUT_SECONDS", "120"))
            ),
            summarization_max_summary_chars=max(
                100, int(os.getenv("GTA_AI_SUMMARIZATION_MAX_SUMMARY_CHARS", "800"))
            ),
            summarization_max_context_tokens=max(
                128,
                int(os.getenv("GTA_AI_SUMMARIZATION_MAX_CONTEXT_TOKENS", "1200")),
            ),
            summarization_recent_turns=max(
                0, int(os.getenv("GTA_AI_SUMMARIZATION_RECENT_TURNS", "2"))
            ),
            redis_key_prefix=os.getenv("GTA_AI_REDIS_KEY_PREFIX", cls.redis_key_prefix).strip(),
            redis_ttl_seconds=max(60, int(os.getenv("GTA_AI_REDIS_TTL_SECONDS", "2592000"))),
            redis_socket_timeout_seconds=max(
                0.1, float(os.getenv("GTA_AI_REDIS_SOCKET_TIMEOUT_SECONDS", "2"))
            ),
            redis_cluster_nodes=tuple(
                value.strip().rstrip("/")
                for value in os.getenv(
                    "GTA_AI_REDIS_CLUSTER_NODES",
                    ",".join(cls.redis_cluster_nodes),
                ).split(",")
                if value.strip()
            ),
            redis_password=os.getenv("GTA_AI_REDIS_PASSWORD", cls.redis_password),
            simple_chat_enabled=os.getenv("GTA_AI_SIMPLE_CHAT_ENABLED", "false").lower()
            in {"1", "true", "yes"},
            simple_chat_url=os.getenv("GTA_AI_SIMPLE_CHAT_URL", cls.simple_chat_url).rstrip("/"),
            simple_chat_model=os.getenv("GTA_AI_SIMPLE_CHAT_MODEL", cls.simple_chat_model),
            simple_chat_max_tokens=max(32, int(os.getenv("GTA_AI_SIMPLE_CHAT_MAX_TOKENS", "192"))),
        )
