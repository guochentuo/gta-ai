"""CPU小模型驱动的异步会话摘要。"""

from __future__ import annotations

import asyncio
import json
import re
import time
from dataclasses import dataclass
from typing import Any

import httpx
import redis

from .prompt_config import PROMPTS
from .services.context_classification_service import (
    TOPICS,
    classify_context,
    extract_profile_facts,
    is_confirmation_followup,
    is_global_summary_query,
    is_user_question_history_query,
    requests_place_recall,
    requests_user_display_name,
    topics_compatible,
)

SUMMARY_SYSTEM_PROMPT = PROMPTS.summarization


@dataclass(frozen=True)
class SummarizationConfig:
    enabled: bool = False
    url: str = "http://127.0.0.1:7107/v1/chat/completions"
    model: str = "Qwen3-4B-Instruct-2507-Q4_K_M"
    timeout_seconds: float = 120.0
    max_summary_chars: int = 800
    max_context_tokens: int = 1200
    recent_turns: int = 2
    redis_key_prefix: str = "gta:ai:session"
    redis_ttl_seconds: int = 2_592_000
    redis_socket_timeout_seconds: float = 2.0
    redis_cluster_nodes: tuple[str, ...] = (
        "192.168.80.4:7001",
        "192.168.80.5:7001",
        "192.168.80.6:7001",
    )
    redis_password: str = ""


class ConversationSummarizer:
    """持久化摘要; 在回答完成后通过CPU模型异步更新。"""

    def __init__(
        self,
        config: SummarizationConfig,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
        redis_client: Any | None = None,
    ) -> None:
        self.config = config
        self.transport = transport
        self._session_locks: dict[str, asyncio.Lock] = {}
        self._redis: Any | None = None
        if config.enabled:
            self._redis = redis_client or _redis_cluster_client(config)

    def get(self, session_id: str, current_query: str = "") -> str:
        if not self.config.enabled or not session_id:
            return ""
        summary, pending = self._snapshot(session_id)
        active_topic = self._active_topic(session_id)
        global_summary_requested = is_global_summary_query(current_query)
        question_history_requested = is_user_question_history_query(current_query)
        if question_history_requested:
            questions = self._questions(session_id) or [
                str(turn.get("user") or "").strip()
                for turn in pending
                if str(turn.get("user") or "").strip()
            ]
            if not questions:
                return ""
            question_memory = "【本会话用户原始问题，按发送顺序】\n" + "\n".join(
                f"{index}. {question}"
                for index, question in enumerate(questions, start=1)
            )
            return _truncate_to_tokens(
                question_memory,
                min(2400, self.config.max_context_tokens * 2),
            )
        if is_confirmation_followup(current_query) and pending:
            pending_topic = str(pending[-1].get("topic") or "")
            if pending_topic in TOPICS:
                active_topic = pending_topic
        current_topic = classify_context(current_query, active_topic)
        topic_summaries = self._topic_summaries(session_id)
        if global_summary_requested:
            unique_summaries = list(dict.fromkeys(topic_summaries.values()))
            summary = "\n\n".join(unique_summaries) or summary
        else:
            summary = topic_summaries.get(
                current_topic,
                (
                    summary
                    if not active_topic or topics_compatible(current_topic, active_topic)
                    else ""
                ),
            )
        summary = _remove_model_generated_profile_claims(summary)
        if not active_topic and current_topic not in {"trip", "quote"}:
            # 旧版Redis只有一份旅游摘要；迁移后的首个非旅游问题不能继承它。
            summary = ""
        profile_facts = self._profile_facts(session_id)
        for turn in pending:
            profile_facts.update(extract_profile_facts(str(turn.get("user") or "")))
        if profile_facts and profile_facts != self._profile_facts(session_id):
            self._store_profile_facts(session_id, profile_facts)
        if profile_facts.get("display_name"):
            current_name = profile_facts["display_name"]
            pending = [
                turn
                for turn in pending
                if not (turn_facts := extract_profile_facts(str(turn.get("user") or "")))
                or turn_facts.get("display_name") == current_name
            ]
            pending = [
                {
                    **turn,
                    "assistant": (
                        ""
                        if _PROFILE_CONFLICT_ASSISTANT_PATTERN.search(
                            str(turn.get("assistant") or "")
                        )
                        else str(turn.get("assistant") or "")
                    ),
                }
                for turn in pending
            ]
        if not global_summary_requested:
            pending = [
                turn
                for turn in pending
                if topics_compatible(current_topic, str(turn.get("topic") or active_topic))
            ]
        places = (
            self._places(session_id)
            if global_summary_requested
            or current_topic in {"trip", "quote"}
            or requests_place_recall(current_query)
            else []
        )
        return _bounded_memory_context(
            summary,
            pending,
            places=places,
            profile_facts=(
                profile_facts
                if global_summary_requested or requests_user_display_name(current_query)
                else None
            ),
            current_query=current_query,
            max_tokens=(
                min(2400, self.config.max_context_tokens * 2)
                if global_summary_requested
                else self.config.max_context_tokens
            ),
            recent_turns=self.config.recent_turns,
        )

    def _snapshot(self, session_id: str) -> tuple[str, list[dict[str, str]]]:
        key = self._redis_key(session_id)
        values = self._redis.hmget(key, "summary", "pending")
        summary = str(values[0] or "")[: self.config.max_summary_chars]
        return summary, _read_pending(values[1])

    def record_turn(self, session_id: str, user_text: str, assistant_text: str) -> None:
        if not self.config.enabled or not session_id or not user_text or not assistant_text:
            return
        active_topic = self._active_topic(session_id)
        topic = classify_context(user_text, active_topic)
        turn = {
            "user": user_text[-2000:],
            "assistant": assistant_text[-3000:],
            "topic": topic,
        }
        key = self._redis_key(session_id)
        while True:
            with self._redis.pipeline(transaction=True) as pipe:
                try:
                    pipe.watch(key)
                    pending = _read_pending(pipe.hget(key, "pending"))
                    questions = _read_questions(pipe.hget(key, "user_questions"))
                    if not questions:
                        questions = [
                            str(item.get("user") or "").strip()
                            for item in pending
                            if str(item.get("user") or "").strip()
                        ]
                    profile_facts = _read_profile_facts(pipe.hget(key, "profile_facts"))
                    profile_facts.update(extract_profile_facts(user_text))
                    pending.append(turn)
                    questions.append(user_text[-1000:])
                    pipe.multi()
                    pipe.hset(
                        key,
                        mapping={
                            "pending": json.dumps(pending[-20:], ensure_ascii=False),
                            "updated_at": str(time.time()),
                            "active_topic": topic,
                            "profile_facts": json.dumps(profile_facts, ensure_ascii=False),
                            "user_questions": json.dumps(questions[-100:], ensure_ascii=False),
                        },
                    )
                    pipe.expire(key, self.config.redis_ttl_seconds)
                    pipe.execute()
                    return
                except redis.WatchError:
                    continue

    def remember_places(self, session_id: str, places: tuple[str, ...]) -> None:
        if not self.config.enabled or not session_id or not places:
            return
        key = self._redis_key(session_id)
        while True:
            with self._redis.pipeline(transaction=True) as pipe:
                try:
                    pipe.watch(key)
                    merged = _read_places(pipe.hget(key, "places"))
                    for place in places:
                        if not place:
                            continue
                        if place in merged:
                            merged.remove(place)
                        merged.append(place)
                    pipe.multi()
                    pipe.hset(
                        key,
                        mapping={
                            "places": json.dumps(merged, ensure_ascii=False),
                            "updated_at": str(time.time()),
                        },
                    )
                    pipe.expire(key, self.config.redis_ttl_seconds)
                    pipe.execute()
                    return
                except redis.WatchError:
                    continue

    def _places(self, session_id: str) -> list[str]:
        return _read_places(self._redis.hget(self._redis_key(session_id), "places"))

    def _profile_facts(self, session_id: str) -> dict[str, str]:
        return _read_profile_facts(self._redis.hget(self._redis_key(session_id), "profile_facts"))

    def _questions(self, session_id: str) -> list[str]:
        return _read_questions(self._redis.hget(self._redis_key(session_id), "user_questions"))

    def contact_was_shown(self, session_id: str) -> bool:
        if not self.config.enabled or not session_id:
            return False
        return str(self._redis.hget(self._redis_key(session_id), "contact_shown") or "") == "1"

    def mark_contact_shown(self, session_id: str) -> bool:
        """首次标记返回True; 同一会话后续调用返回False。"""
        if not self.config.enabled or not session_id:
            return True
        key = self._redis_key(session_id)
        inserted = bool(self._redis.hsetnx(key, "contact_shown", "1"))
        self._redis.expire(key, self.config.redis_ttl_seconds)
        return inserted

    def _save(self, session_id: str, summary: str, consumed_turns: int, topic: str) -> None:
        summary = summary.strip()[: self.config.max_summary_chars]
        if not summary:
            return
        key = self._redis_key(session_id)
        while True:
            with self._redis.pipeline(transaction=True) as pipe:
                try:
                    pipe.watch(key)
                    pending = _read_pending(pipe.hget(key, "pending"))
                    topic_summaries = _read_topic_summaries(pipe.hget(key, "topic_summaries"))
                    topic_summaries[topic] = summary
                    snapshot_size = min(consumed_turns, len(pending))
                    matching_indices = [
                        index
                        for index, turn in enumerate(pending[:snapshot_size])
                        if str(turn.get("topic") or "") == topic
                    ]
                    retained_count = min(self.config.recent_turns, len(matching_indices))
                    removable_count = len(matching_indices) - retained_count
                    removable = set(matching_indices[:removable_count])
                    remaining = [
                        turn for index, turn in enumerate(pending) if index not in removable
                    ]
                    pipe.multi()
                    pipe.hset(
                        key,
                        mapping={
                            "summary": summary,
                            "pending": json.dumps(remaining, ensure_ascii=False),
                            "updated_at": str(time.time()),
                            "active_topic": topic,
                            "topic_summaries": json.dumps(topic_summaries, ensure_ascii=False),
                        },
                    )
                    pipe.expire(key, self.config.redis_ttl_seconds)
                    pipe.execute()
                    return
                except redis.WatchError:
                    continue

    async def update(
        self,
        session_id: str,
        user_text: str = "",
        assistant_text: str = "",
    ) -> str:
        if not self.config.enabled or not session_id:
            return ""
        if user_text and assistant_text:
            self.record_turn(session_id, user_text, assistant_text)
        lock = self._session_locks.setdefault(session_id, asyncio.Lock())
        async with lock:
            distributed_lock = (
                self._redis.lock(
                    self._redis_key(session_id) + ":summary-lock",
                    timeout=max(5, round(self.config.timeout_seconds + 10)),
                    blocking_timeout=0.1,
                )
                if self._redis is not None
                else None
            )
            if distributed_lock is not None and not distributed_lock.acquire(blocking=True):
                return ""
            previous, pending = self._snapshot(session_id)
            if not pending:
                if distributed_lock is not None:
                    distributed_lock.release()
                return previous
            fallback_topic = str(pending[-1].get("topic") or "general")
            topic_pending = [
                turn
                for turn in pending
                if str(turn.get("topic") or "general") == fallback_topic
            ]
            previous = self._topic_summaries(session_id).get(fallback_topic, previous)
            content = json.dumps(
                {
                    "已有摘要": previous,
                    "待合并对话": topic_pending,
                },
                ensure_ascii=False,
            )
            payload = {
                "model": self.config.model,
                "messages": [
                    {"role": "system", "content": SUMMARY_SYSTEM_PROMPT},
                    {"role": "user", "content": content},
                ],
                "temperature": 0.1,
                "max_tokens": 800,
                "stream": False,
            }
            try:
                timeout = httpx.Timeout(self.config.timeout_seconds, connect=2)
                async with httpx.AsyncClient(timeout=timeout, transport=self.transport) as client:
                    response = await client.post(self.config.url, json=payload)
                    response.raise_for_status()
                model_output = response.json()["choices"][0]["message"]["content"]
                if not isinstance(model_output, str):
                    raise ValueError("摘要模型返回内容不是字符串")
                _, summary = _read_classified_summary(model_output, fallback_topic)
                summary = _remove_model_generated_profile_claims(summary)
                self._save(session_id, summary, len(pending), fallback_topic)
                return summary.strip()
            finally:
                if distributed_lock is not None and distributed_lock.owned():
                    distributed_lock.release()

    def _redis_key(self, session_id: str) -> str:
        return f"{self.config.redis_key_prefix}:{{{session_id}}}"

    def _active_topic(self, session_id: str) -> str:
        value = str(self._redis.hget(self._redis_key(session_id), "active_topic") or "")
        return value if value in TOPICS else ""

    def _topic_summaries(self, session_id: str) -> dict[str, str]:
        return _read_topic_summaries(
            self._redis.hget(self._redis_key(session_id), "topic_summaries")
        )

    def _store_profile_facts(self, session_id: str, facts: dict[str, str]) -> None:
        key = self._redis_key(session_id)
        self._redis.hset(
            key,
            mapping={
                "profile_facts": json.dumps(facts, ensure_ascii=False),
                "updated_at": str(time.time()),
            },
        )
        self._redis.expire(key, self.config.redis_ttl_seconds)


def _read_pending(value: Any) -> list[dict[str, str]]:
    if value is None:
        return []
    try:
        parsed = json.loads(value if isinstance(value, str) else value.decode())
    except (json.JSONDecodeError, UnicodeDecodeError, AttributeError):
        return []
    if not isinstance(parsed, list):
        return []
    return [
        {
            "user": str(item.get("user", "")),
            "assistant": str(item.get("assistant", "")),
            "topic": str(item.get("topic", "")),
        }
        for item in parsed
        if isinstance(item, dict)
    ]


def _read_questions(value: Any) -> list[str]:
    if value is None:
        return []
    try:
        parsed = json.loads(value if isinstance(value, str) else value.decode())
    except (json.JSONDecodeError, UnicodeDecodeError, AttributeError):
        return []
    if not isinstance(parsed, list):
        return []
    return [str(item).strip() for item in parsed if str(item).strip()]


def _read_classified_summary(value: str, fallback_topic: str) -> tuple[str, str]:
    try:
        payload = json.loads(value)
    except json.JSONDecodeError:
        return fallback_topic, value
    if not isinstance(payload, dict):
        return fallback_topic, value
    topic = str(payload.get("topic") or fallback_topic)
    summary = str(payload.get("summary") or "").strip()
    return (topic if topic in TOPICS else fallback_topic), (summary or value)


_MODEL_PROFILE_CLAIM_PATTERN = re.compile(
    r"(?:^|(?<=[。！？\n]))\s*[^。！？\n]{0,80}"
    r"(?:用户身份|用户姓名|用户称呼|用户自称|称呼用户)"
    r"[^。！？\n]{0,80}[。！？]?"
)


def _remove_model_generated_profile_claims(summary: str) -> str:
    """姓名只能来自Router提取的用户原话, 禁止4B从助手回答中生成身份事实。"""
    return _MODEL_PROFILE_CLAIM_PATTERN.sub("", summary).strip()


def _read_topic_summaries(value: Any) -> dict[str, str]:
    if value is None:
        return {}
    try:
        payload = json.loads(value if isinstance(value, str) else value.decode())
    except (json.JSONDecodeError, UnicodeDecodeError, AttributeError):
        return {}
    if not isinstance(payload, dict):
        return {}
    return {
        str(topic): str(summary)
        for topic, summary in payload.items()
        if str(topic) in TOPICS and str(summary).strip()
    }


def _read_places(value: Any) -> list[str]:
    if value is None:
        return []
    try:
        parsed = json.loads(value if isinstance(value, str) else value.decode())
    except (json.JSONDecodeError, UnicodeDecodeError, AttributeError):
        return []
    if not isinstance(parsed, list):
        return []
    return [str(item).strip() for item in parsed if str(item).strip()]


def _read_profile_facts(value: Any) -> dict[str, str]:
    if value is None:
        return {}
    try:
        parsed = json.loads(value if isinstance(value, str) else value.decode())
    except (json.JSONDecodeError, UnicodeDecodeError, AttributeError):
        return {}
    if not isinstance(parsed, dict):
        return {}
    display_name = str(parsed.get("display_name") or "").strip()
    return {"display_name": display_name} if display_name else {}


_WORD_PATTERN = re.compile(r"[A-Za-z0-9]{2,}|[\u4e00-\u9fff]{2,}")
_PROFILE_CONFLICT_ASSISTANT_PATTERN = re.compile(
    r"无法获取.{0,30}(?:隐私|身份)|不知道.{0,20}(?:身份|称呼)|无法识别.{0,20}身份"
)


def estimate_tokens(text: str) -> int:
    """Estimate token usage without loading the 27B tokenizer."""
    cjk = sum("\u4e00" <= char <= "\u9fff" for char in text)
    other = len(text) - cjk
    return cjk + (other + 3) // 4


def _query_terms(text: str) -> set[str]:
    terms: set[str] = set()
    for value in _WORD_PATTERN.findall(text.lower()):
        if any("\u4e00" <= char <= "\u9fff" for char in value):
            terms.update(value[index : index + 2] for index in range(len(value) - 1))
        else:
            terms.add(value)
    return terms


def _bounded_memory_context(
    summary: str,
    pending: list[dict[str, str]],
    *,
    places: list[str] | None = None,
    profile_facts: dict[str, str] | None = None,
    current_query: str,
    max_tokens: int,
    recent_turns: int,
) -> str:
    """在固定预算内保留摘要、相关轮次和少量最近轮次。"""
    if max_tokens <= 0:
        return ""
    query_terms = _query_terms(current_query)
    candidates: list[tuple[int, int, int, str]] = []
    recent_start = max(0, len(pending) - max(0, recent_turns))
    for index, turn in enumerate(pending):
        text = PROMPTS.memory_turn_context.format(
            user=turn.get("user", ""),
            assistant=turn.get("assistant", ""),
        ).strip()
        overlap = len(query_terms & _query_terms(text))
        is_recent = index >= recent_start
        if overlap or is_recent:
            candidates.append((1 if is_recent else 0, overlap, index, text))

    # 摘要是已经去除寒暄后的长期事实，优先保留；超限时截断尾部。
    parts: list[str] = []
    used = 0
    if profile_facts and profile_facts.get("display_name"):
        display_name = profile_facts["display_name"]
        profile_memory = PROMPTS.memory_profile_context.format(display_name=display_name)
        bounded_profile = _truncate_to_tokens(profile_memory, max(1, max_tokens // 4))
        parts.append(bounded_profile)
        used += estimate_tokens(bounded_profile)
    if places:
        place_memory = PROMPTS.memory_places_context.format(places="、".join(places))
        bounded_places = _truncate_to_tokens(place_memory, max(1, max_tokens // 4))
        if bounded_places:
            parts.append(bounded_places)
            used += estimate_tokens(bounded_places)
    if summary:
        summary_budget = max(1, min(max_tokens * 2 // 3, max_tokens - used))
        bounded_summary = _truncate_to_tokens(summary, summary_budget)
        if bounded_summary:
            parts.append(bounded_summary)
            used += estimate_tokens(bounded_summary)

    selected: list[str] = []
    # 先放与当前问题更相关的内容；同分时最近轮次优先。
    for _, _, _, text in sorted(
        candidates,
        key=lambda item: (item[1], item[0], item[2]),
        reverse=True,
    ):
        remaining = max_tokens - used
        if remaining <= 8:
            break
        bounded = _truncate_to_tokens(text, remaining)
        if bounded:
            selected.append(bounded)
            used += estimate_tokens(bounded)
    if selected:
        parts.append(PROMPTS.memory_pending_context.format(turns="\n".join(selected)))
    return _truncate_to_tokens("\n\n".join(parts), max_tokens)


def _truncate_to_tokens(text: str, max_tokens: int) -> str:
    if estimate_tokens(text) <= max_tokens:
        return text
    low, high = 0, len(text)
    while low < high:
        middle = (low + high + 1) // 2
        if estimate_tokens(text[:middle]) <= max_tokens:
            low = middle
        else:
            high = middle - 1
    return text[:low].rstrip()


def _redis_cluster_client(config: SummarizationConfig) -> Any:
    if not config.redis_cluster_nodes:
        raise RuntimeError("Redis集群节点不能为空")
    host, port = config.redis_cluster_nodes[0].rsplit(":", 1)
    common: dict[str, Any] = {
        "password": config.redis_password or None,
        "decode_responses": True,
        "socket_connect_timeout": config.redis_socket_timeout_seconds,
        "socket_timeout": config.redis_socket_timeout_seconds,
        "health_check_interval": 30,
    }
    client: Any = redis.RedisCluster(host=host, port=int(port), **common)
    client.ping()
    return client
