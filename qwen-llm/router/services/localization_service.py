# ruff: noqa: RUF002
"""多语言回复规则与人工服务事件。

这些内容属于界面本地化和回复策略，不属于模型知识，也不应堆在FastAPI入口中。
"""

from __future__ import annotations

import json
import re

import httpx

from ..prompt_config import PROMPTS

CONTACT_EXPLICIT_PATTERN = re.compile(
    r"怎么联系|如何联系|联系方式|联系顾问|顾问.{0,8}联系|"
    r"客服电话|客服微信|加微信|WhatsApp|LINE"
)
LANGUAGE_MATCH_PROMPT = PROMPTS.language_match


_LANGUAGE_OUTPUT_RULES = PROMPTS.language_output_rules
_LANGUAGE_NAMES = PROMPTS.language_names

# 只有真正出现简繁差异字时才覆盖调用方地区提示。没有差异字的“杭州行程”仍可
# 依据zh-HK/zh-TW选择繁体, 明确写了“绿、网、们、务”等简体字时必须保持简体。
_SIMPLIFIED_CHINESE_PATTERN = re.compile(
    r"[简体台湾联系顾问务选择这里还与为会个们游费价车饭馆门开后发么来时说"
    r"对请应该点线网页资预订划画观间将让从无绿绍属谁]"
)
_TRADITIONAL_CHINESE_PATTERN = re.compile(
    r"[體臺灣聯繫顧問務選擇這裡還與為爲會個們遊費價車飯館門開後發麼麽來時"
    r"說對請應該點線網頁資預訂劃畫觀間將讓從無綠紹屬誰]"
)


def _language_output_rule(locale: str) -> str:
    if locale in _LANGUAGE_OUTPUT_RULES:
        return _LANGUAGE_OUTPUT_RULES[locale]
    language = _LANGUAGE_NAMES.get(locale, "English")
    return f"IMPORTANT: Write the entire response in natural {language} only."


def language_style_prompt(
    user_text: str = "", locale_hint: str = "", country_hint: str = ""
) -> str:
    locale = locale_hint.strip() if re.fullmatch(r"[A-Za-z0-9_-]{2,20}", locale_hint) else ""
    country = country_hint.strip() if re.fullmatch(r"[A-Za-z]{2,3}", country_hint) else ""
    output_rule = _language_output_rule(_interface_locale(user_text, locale_hint))
    if not locale and not country:
        return LANGUAGE_MATCH_PROMPT + output_rule
    return (
        LANGUAGE_MATCH_PROMPT
        + output_rule
        + PROMPTS.locale_context.format(
            locale=locale or "unknown",
            country=country.upper() or "unknown",
        )
    )


def query_explicitly_requests_contact(query: str) -> bool:
    return bool(CONTACT_EXPLICIT_PATTERN.search(query.strip()))


def _interface_locale(text: str, locale_hint: str = "") -> str:
    normalized_hint = locale_hint.replace("_", "-").lower()
    if re.search(r"[\u3040-\u30ff]", text):
        return "ja"
    if re.search(r"[\uac00-\ud7af]", text):
        return "ko"
    if re.search(r"[\u0600-\u06ff]", text):
        return "ar"
    if re.search(r"[\u0590-\u05ff]", text):
        return "he"
    if re.search(r"[\u0900-\u097f]", text):
        return "hi"
    if re.search(r"[\u0e00-\u0e7f]", text):
        return "th"
    if re.search(r"[\u0400-\u04ff]", text):
        return "uk" if re.search(r"[іїєґІЇЄҐ]", text) else "ru"
    traditional_count = len(_TRADITIONAL_CHINESE_PATTERN.findall(text))
    simplified_count = len(_SIMPLIFIED_CHINESE_PATTERN.findall(text))
    if traditional_count or simplified_count:
        return "zh_tw" if traditional_count > simplified_count else "zh_cn"
    if re.search(r"[\u4e00-\u9fff]", text):
        if normalized_hint.startswith(("zh-tw", "zh-hk", "zh-mo", "zh-hant")):
            return "zh_tw"
        return "zh_cn"
    if normalized_hint.startswith("de"):
        return "de"
    if re.search(
        r"\b(?:wie|was|wann|wo|warum|reise|reisen|urlaub|kostet|tage|hotel|"
        r"sehensw[uü]rdigkeit(?:en)?|empfehl(?:en|ung))\b",
        text,
        re.IGNORECASE,
    ):
        return "de"
    latin_language_patterns = (
        ("fr", r"\b(?:voyage|voyager|combien|bonjour|séjour|visiter|itinéraire)\b"),
        ("es", r"\b(?:viaje|viajar|cuánto|hola|visitar|itinerario|turismo)\b"),
        ("pt", r"\b(?:viagem|viajar|quanto|olá|visitar|roteiro|turismo)\b"),
        ("it", r"\b(?:viaggio|viaggiare|quanto|ciao|visitare|itinerario|turismo)\b"),
        ("nl", r"\b(?:reis|reizen|hoeveel|hallo|bezoeken|route|vakantie)\b"),
        ("pl", r"\b(?:podróż|podroz|ile|cześć|zwiedzić|wycieczka|wakacje)\b"),
        ("tr", r"\b(?:seyahat|gezi|kaç|merhaba|ziyaret|tatil|turizm)\b"),
        ("vi", r"\b(?:du lịch|chuyến đi|bao nhiêu|xin chào|tham quan)\b"),
        ("id", r"\b(?:perjalanan|berapa|halo|wisata|mengunjungi|liburan)\b"),
        ("ms", r"\b(?:perjalanan|berapa|hai|pelancongan|melawat|percutian)\b"),
    )
    for language, pattern in latin_language_patterns:
        if re.search(pattern, text, re.IGNORECASE):
            return language
    primary_hint = normalized_hint.split("-", 1)[0]
    if primary_hint in _LANGUAGE_NAMES:
        return primary_hint
    if re.search(r"[A-Za-z]", text):
        return "en"
    if normalized_hint.startswith(("zh-tw", "zh-hk", "zh-mo", "zh-hant")):
        return "zh_tw"
    if normalized_hint.startswith("zh") or not text.strip():
        return "zh_cn"
    if normalized_hint.startswith("ja"):
        return "ja"
    if normalized_hint.startswith("ko"):
        return "ko"
    return "en"


async def generate_handoff_offer(
    user_text: str,
    *,
    locale_hint: str,
    country_hint: str,
    url: str,
    model: str,
    timeout_seconds: float,
    transport: httpx.AsyncBaseTransport | None = None,
) -> dict[str, str]:
    """由27B按当前用户语言生成转人工提示，Router只校验结构。"""
    request_context = (
        f"locale={locale_hint or 'unknown'}\n"
        f"country={country_hint or 'unknown'}\n"
        f"current_user_message={user_text}"
    )
    async with httpx.AsyncClient(
        timeout=httpx.Timeout(timeout_seconds), transport=transport
    ) as client:
        response = await client.post(
            url,
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": PROMPTS.handoff_localization},
                    {"role": "user", "content": request_context},
                ],
                "stream": False,
                "temperature": 0.1,
                "max_tokens": 96,
                "chat_template_kwargs": {"enable_thinking": False},
            },
        )
        response.raise_for_status()
        raw = str(response.json().get("choices", [{}])[0].get("message", {}).get("content", ""))
    start = raw.find("{")
    end = raw.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("27B没有返回人工服务文案JSON")
    payload = json.loads(raw[start : end + 1])
    prompt = str(payload.get("prompt", "")).strip()
    action_label = str(payload.get("action_label", "")).strip()
    if not prompt or not action_label or len(prompt) > 160 or len(action_label) > 48:
        raise ValueError("27B返回的人工服务文案无效")
    return {"prompt": prompt, "action_label": action_label}


def contact_request_event() -> bytes:
    """只通知调用方用户主动请求人工，不携带联系方式或界面文案。"""
    return b'data: {"type":"contact_request"}\n\n'


def handoff_offer_event(copy: dict[str, str]) -> bytes:
    """把27B生成的本地化文案封装为稳定的SSE事件。"""
    payload = {
        "type": "handoff_offer",
        "prompt": copy["prompt"],
        "action_label": copy["action_label"],
    }
    return (
        "data: " + json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n\n"
    ).encode()
