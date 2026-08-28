# ruff: noqa: RUF002
"""欢迎模板按需本地化与Redis缓存。

ES只保存中文母版；目标语言版本由本机27B按需生成，并作为可失效缓存保存。
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any

import httpx
import redis

from .welcome_template_service import WelcomeTemplate

_LOCALE_PATTERN = re.compile(r"^[A-Za-z]{2,3}(?:[-_][A-Za-z0-9]{2,8}){0,2}$")
_INTERNAL_MARKER_PATTERN = re.compile(r"\[\[[A-Z0-9_]+\]\]")
_URL_PATTERN = re.compile(r"https?://[^\s)\]]+")
_CACHE_FORMAT_VERSION = "v3"


@dataclass(frozen=True)
class WelcomeLocalizationConfig:
    enabled: bool = True
    url: str = "http://127.0.0.1:18086/v1/chat/completions"
    model: str = "Qwen/Qwen3.8-27B-FP8"
    timeout_seconds: float = 120.0
    max_tokens: int = 4096
    cache_key_prefix: str = "gta:ai:welcome-localized"
    cache_ttl_seconds: int = 2_592_000
    redis_socket_timeout_seconds: float = 2.0
    redis_cluster_nodes: tuple[str, ...] = ()
    redis_password: str = ""
    prewarm_locales: tuple[str, ...] = ()
    prewarm_keyword: str = "中国旅行"


def normalize_welcome_locale(locale_hint: str, country_hint: str = "") -> str:
    """返回标准化BCP-47语言标签；无法判断时返回中文母版语言。"""
    raw = locale_hint.strip()
    if not raw or not _LOCALE_PATTERN.fullmatch(raw):
        return "zh-CN"
    parts = raw.replace("_", "-").split("-")
    language = parts[0].lower()
    region = ""
    for part in parts[1:]:
        if len(part) == 2 and part.isalpha():
            region = part.upper()
            break
    country = country_hint.strip().upper()
    if language == "zh":
        if region in {"TW", "HK", "MO"} or country in {"TW", "HK", "MO"}:
            return f"zh-{region or country}"
        return "zh-CN"
    return f"{language}-{region}" if region else language


def _is_source_locale(locale: str) -> bool:
    return locale.lower() in {"zh", "zh-cn", "zh-sg", "zh-hans"}


class WelcomeLocalizationService:
    def __init__(
        self,
        config: WelcomeLocalizationConfig,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
        redis_client: Any | None = None,
    ) -> None:
        self.config = config
        self._transport = transport
        self._redis = redis_client
        self._redis_initialized = redis_client is not None

    async def localize(
        self,
        template: WelcomeTemplate,
        *,
        locale_hint: str,
        country_hint: str = "",
    ) -> WelcomeTemplate:
        locale = normalize_welcome_locale(locale_hint, country_hint)
        if not self.config.enabled or _is_source_locale(locale):
            return template

        cache_key = self._cache_key(template, locale, country_hint)
        cached = await self._cache_get(cache_key)
        if cached:
            return self._decode_template(cached, template)

        localized = await self._generate(template, locale, country_hint)
        await self._cache_set(cache_key, self._encode_template(localized))
        return localized

    async def prewarm(
        self,
        template: WelcomeTemplate,
    ) -> list[str]:
        completed: list[str] = []
        for locale in self.config.prewarm_locales:
            normalized = normalize_welcome_locale(locale)
            if _is_source_locale(normalized):
                continue
            try:
                await self.localize(template, locale_hint=normalized)
                completed.append(normalized)
            except (httpx.HTTPError, ValueError, KeyError, redis.RedisError):
                continue
        return completed

    async def _generate(
        self,
        template: WelcomeTemplate,
        locale: str,
        country_hint: str,
    ) -> WelcomeTemplate:
        source = {
            "title": template.title,
            "content": template.content,
            "suggested_questions": list(template.suggested_questions),
            "ui_text": template.ui_text,
        }
        system_prompt = (
            "You are a tourism marketing localization editor. Localize the supplied Chinese "
            "content for the requested locale and country. Write naturally for local travelers; "
            "do not translate literally. Preserve facts, prices, company names, Markdown "
            "structure, every [[UPPER_CASE_MARKER]], and every URL exactly. Do not add "
            "claims. Return one valid JSON object only, with exactly these keys: title, "
            "content, suggested_questions, ui_text. "
            "Keep every ui_text key unchanged and localize only its value."
        )
        user_prompt = json.dumps(
            {
                "target_locale": locale,
                "target_country": country_hint.strip().upper() or "unknown",
                "source_locale": "zh-CN",
                "source": source,
            },
            ensure_ascii=False,
        )
        timeout = httpx.Timeout(self.config.timeout_seconds)
        async with httpx.AsyncClient(timeout=timeout, transport=self._transport) as client:
            response = await client.post(
                self.config.url,
                json={
                    "model": self.config.model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    "stream": False,
                    "temperature": 0.1,
                    "max_tokens": self.config.max_tokens,
                    "chat_template_kwargs": {"enable_thinking": False},
                },
            )
            response.raise_for_status()
            raw_content = (
                response.json().get("choices", [{}])[0].get("message", {}).get("content", "")
            )
        payload = self._parse_json(str(raw_content))
        localized = self._validated_template(payload, template)
        return localized

    @staticmethod
    def _parse_json(value: str) -> dict[str, Any]:
        text = value.strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.IGNORECASE)
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("欢迎模板本地化模型没有返回JSON")
        payload = json.loads(text[start : end + 1])
        if not isinstance(payload, dict):
            raise ValueError("欢迎模板本地化结果必须是对象")
        return payload

    @staticmethod
    def _validated_template(payload: dict[str, Any], source: WelcomeTemplate) -> WelcomeTemplate:
        title = _normalize_brand_names(str(payload.get("title", ""))).strip()
        content = _normalize_brand_names(str(payload.get("content", ""))).strip()
        questions_raw = payload.get("suggested_questions", [])
        ui_raw = payload.get("ui_text", {})
        if (
            not title
            or not content
            or not isinstance(questions_raw, list)
            or not isinstance(ui_raw, dict)
        ):
            raise ValueError("欢迎模板本地化结果字段不完整")
        if set(ui_raw) != set(source.ui_text):
            raise ValueError("欢迎模板本地化结果改变了ui_text字段")
        if sorted(_INTERNAL_MARKER_PATTERN.findall(content)) != sorted(
            _INTERNAL_MARKER_PATTERN.findall(source.content)
        ):
            raise ValueError("欢迎模板本地化结果改变了内部标记")
        if sorted(_URL_PATTERN.findall(content)) != sorted(_URL_PATTERN.findall(source.content)):
            raise ValueError("欢迎模板本地化结果改变了URL")
        questions = tuple(
            _normalize_brand_names(str(item)).strip() for item in questions_raw if str(item).strip()
        )
        if len(questions) != len(source.suggested_questions):
            raise ValueError("欢迎模板本地化结果改变了推荐问题数量")
        ui_text = {
            str(key): _normalize_brand_names(str(value)).strip() for key, value in ui_raw.items()
        }
        if any(not value for value in ui_text.values()):
            raise ValueError("欢迎模板本地化结果包含空界面文字")
        return WelcomeTemplate(
            template_id=source.template_id,
            title=title,
            content=content,
            images=source.images,
            suggested_questions=questions,
            ui_text=ui_text,
            version=source.version,
        )

    def _cache_key(self, template: WelcomeTemplate, locale: str, country_hint: str) -> str:
        identity = (
            f"{_CACHE_FORMAT_VERSION}:{template.template_id}:{template.version}:"
            f"{locale}:{country_hint.upper()}"
        )
        digest = hashlib.sha256(identity.encode()).hexdigest()[:24]
        return f"{self.config.cache_key_prefix}:{{{digest}}}"

    async def _cache_get(self, key: str) -> str:
        client = await self._redis_client()
        if client is None:
            return ""
        try:
            value = await asyncio.to_thread(client.get, key)
        except redis.RedisError:
            return ""
        if isinstance(value, bytes):
            return value.decode("utf-8")
        return str(value or "")

    async def _cache_set(self, key: str, value: str) -> None:
        client = await self._redis_client()
        if client is None:
            return
        try:
            await asyncio.to_thread(client.setex, key, self.config.cache_ttl_seconds, value)
        except redis.RedisError:
            return

    async def _redis_client(self) -> Any | None:
        if self._redis_initialized:
            return self._redis
        self._redis_initialized = True
        if not self.config.redis_cluster_nodes:
            return None
        host, port = self.config.redis_cluster_nodes[0].rsplit(":", 1)
        self._redis = redis.RedisCluster(
            host=host,
            port=int(port),
            password=self.config.redis_password or None,
            decode_responses=True,
            socket_connect_timeout=self.config.redis_socket_timeout_seconds,
            socket_timeout=self.config.redis_socket_timeout_seconds,
            skip_full_coverage_check=True,
        )
        return self._redis

    @staticmethod
    def _encode_template(template: WelcomeTemplate) -> str:
        return json.dumps(
            {
                "title": template.title,
                "content": template.content,
                "suggested_questions": list(template.suggested_questions),
                "ui_text": template.ui_text,
            },
            ensure_ascii=False,
        )

    @staticmethod
    def _decode_template(value: str, source: WelcomeTemplate) -> WelcomeTemplate:
        payload = json.loads(value)
        return WelcomeLocalizationService._validated_template(payload, source)


def _normalize_brand_names(value: str) -> str:
    def normalize_text(text: str) -> str:
        normalized = re.sub(r"\bGreen\s*Tour\s*AI\b", "GreenTourAI", text, flags=re.IGNORECASE)
        return re.sub(r"\bGreen\s*Tour\s*Asia\b", "GreenTourAsia", normalized, flags=re.IGNORECASE)

    parts: list[str] = []
    cursor = 0
    for match in _URL_PATTERN.finditer(value):
        parts.append(normalize_text(value[cursor : match.start()]))
        parts.append(match.group(0))
        cursor = match.end()
    parts.append(normalize_text(value[cursor:]))
    return "".join(parts)


def welcome_handoff_payload(ui_text: dict[str, str]) -> dict[str, Any]:
    """使用已本地化的ES界面文字构造人工服务事件, 账号不交给模型生成。"""
    return {
        "type": "handoff_offer",
        "title": ui_text.get("handoff_title", ""),
        "description": ui_text.get("handoff_description", ""),
        "contacts": [
            {
                "channel": "WhatsApp",
                "account": "+852 5555 8888",
                "url": "https://wa.me/85255558888",
            },
            {
                "channel": "LINE",
                "account": "@greentourasia",
                "url": "https://line.me/R/ti/p/@greentourasia",
            },
            {
                "channel": "WeChat",
                "account": "GreenTourAsia",
                "url": "",
            },
        ],
        "prompt": ui_text.get("handoff_prompt", ""),
        "action_label": ui_text.get("handoff_action_label", ""),
    }


def welcome_handoff_event(ui_text: dict[str, str]) -> bytes:
    payload = json.dumps(
        welcome_handoff_payload(ui_text),
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return f"data: {payload}\n\n".encode()
