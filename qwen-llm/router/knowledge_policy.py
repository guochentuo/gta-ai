from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path

DEFAULT_POLICY_PATH = Path(__file__).resolve().parents[1] / "config" / "knowledge-policy.toml"


@dataclass(frozen=True)
class KnowledgePolicy:
    simple_chat_prompt: str
    tool_name: str
    tool_description: str
    tool_query_description: str
    direct_terms: tuple[str, ...]
    international_terms: tuple[str, ...]
    business_entities: tuple[str, ...]
    complex_terms: tuple[str, ...]
    conversation_topics: tuple[str, ...]
    invalid_display_names: tuple[str, ...]
    identity_question_terms: tuple[str, ...]
    site_about_terms: tuple[str, ...]
    site_help_terms: tuple[str, ...]
    labels: dict[str, str]


def load_knowledge_policy(path: Path = DEFAULT_POLICY_PATH) -> KnowledgePolicy:
    with path.open("rb") as policy_file:
        payload = tomllib.load(policy_file)
    simple_chat = payload["simple_chat"]
    tool = payload["knowledge_tool"]
    retrieval = payload["retrieval"]
    conversation = payload["conversation"]
    labels = payload["knowledge_labels"]
    return KnowledgePolicy(
        simple_chat_prompt=str(simple_chat["prompt"]).strip(),
        tool_name=str(tool["name"]).strip(),
        tool_description=str(tool["description"]).strip(),
        tool_query_description=str(tool["query_description"]).strip(),
        direct_terms=tuple(str(value) for value in retrieval["direct_terms"]),
        international_terms=tuple(str(value) for value in retrieval["international_terms"]),
        business_entities=tuple(str(value) for value in retrieval["business_entities"]),
        complex_terms=tuple(str(value) for value in retrieval["complex_terms"]),
        conversation_topics=tuple(str(value) for value in conversation["topics"]),
        invalid_display_names=tuple(
            str(value) for value in conversation["invalid_display_names"]
        ),
        identity_question_terms=tuple(
            str(value) for value in conversation["identity_question_terms"]
        ),
        site_about_terms=tuple(str(value) for value in conversation["site_about_terms"]),
        site_help_terms=tuple(str(value) for value in conversation["site_help_terms"]),
        labels={str(key): str(value) for key, value in labels.items()},
    )
