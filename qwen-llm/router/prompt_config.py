"""集中加载模型提示词和可运营文案。"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path

DEFAULT_PROMPTS_PATH = Path(__file__).resolve().parents[1] / "config" / "prompts.toml"


@dataclass(frozen=True)
class PromptCatalog:
    persona: str
    retrieval: str
    company_recommendation: str
    verified_business_fact: str
    confirmation_followup: str
    travel_support: str
    playful_query: str
    question_history: str
    language_match: str
    locale_context: str
    language_output_rules: dict[str, str]
    language_names: dict[str, str]
    retrieval_query_normalization: str
    memory_authoritative_context: str
    memory_profile_context: str
    memory_places_context: str
    memory_pending_context: str
    memory_turn_context: str
    summarization: str
    welcome_localization: str
    handoff_localization: str
    image_marker_context: str
    image_markdown_context: str
    pricing_missing: str
    pricing_unavailable: str
    pricing_quote_footer: str
    pricing_minimum_group: str


def load_prompt_catalog(path: Path = DEFAULT_PROMPTS_PATH) -> PromptCatalog:
    with path.open("rb") as prompt_file:
        payload = tomllib.load(prompt_file)

    def text(section: str, key: str) -> str:
        value = str(payload[section][key]).strip()
        if not value:
            raise ValueError(f"提示词配置不能为空: {section}.{key}")
        return value

    def text_map(section: str) -> dict[str, str]:
        values = {str(key): str(value).strip() for key, value in payload[section].items()}
        if not values or any(not value for value in values.values()):
            raise ValueError(f"提示词映射配置不能为空: {section}")
        return values

    return PromptCatalog(
        persona=text("persona", "system"),
        retrieval=text("retrieval", "context"),
        company_recommendation=text("company_recommendation", "context"),
        verified_business_fact=text("verified_business_fact", "context"),
        confirmation_followup=text("confirmation_followup", "context"),
        travel_support=text("travel_support", "context"),
        playful_query=text("playful_query", "context"),
        question_history=text("question_history", "context"),
        language_match=text("language_match", "context"),
        locale_context=text("language_match", "locale_context"),
        language_output_rules=text_map("language_output_rules"),
        language_names=text_map("language_names"),
        retrieval_query_normalization=text("retrieval_query_normalization", "system"),
        memory_authoritative_context=text("memory", "authoritative_context"),
        memory_profile_context=text("memory", "profile_context"),
        memory_places_context=text("memory", "places_context"),
        memory_pending_context=text("memory", "pending_context"),
        memory_turn_context=text("memory", "turn_context"),
        summarization=text("summarization", "system"),
        welcome_localization=text("welcome_localization", "system"),
        handoff_localization=text("handoff_localization", "system"),
        image_marker_context=text("image_context", "marker"),
        image_markdown_context=text("image_context", "markdown"),
        pricing_missing=text("pricing", "missing"),
        pricing_unavailable=text("pricing", "unavailable"),
        pricing_quote_footer=text("pricing", "quote_footer"),
        pricing_minimum_group=text("pricing", "minimum_group"),
    )


PROMPTS = load_prompt_catalog()
