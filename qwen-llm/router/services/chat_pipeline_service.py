# ruff: noqa: F401, RUF001
"""聊天请求转换、知识上下文和流式响应过滤服务。"""

from __future__ import annotations

import json
import re
import tomllib
from pathlib import Path

import httpx

from ..knowledge_policy import load_knowledge_policy
from ..retrieval import RetrievalResult
from .localization_service import (
    EXPLICIT_CONTACT_PROMPT,
    LANGUAGE_MATCH_PROMPT,
    contact_text_event,
    handoff_offer_event,
    language_style_prompt,
    query_explicitly_requests_contact,
)

KNOWLEDGE_POLICY = load_knowledge_policy()
SIMPLE_CHAT_PROMPT = KNOWLEDGE_POLICY.simple_chat_prompt
HISTORICAL_IDENTITY_INTRO_PATTERN = re.compile(
    r"^\s*"
    r"(?:[^。！？!?.\n]{0,30}[。！？!?.]\s*)?"
    r"(?=[^。！？!?.\n]{0,180}(?:小梦|xiao\s*meng|xiaomeng))"
    r"[^。！？!?.\n]{0,180}[。！？!?.]\s*",
    re.IGNORECASE,
)
IDENTITY_QUESTION_TERMS = (
    "你是谁",
    "你叫什么",
    "你的名字",
    "介绍一下你自己",
    "自我介绍",
    "哪个公司",
    "哪家公司",
    "所属公司",
    "属于什么公司",
    "谁开发",
    "谁研发",
    "谁运营",
    "谁创建",
    "开发者是谁",
    "研发方",
    "运营方",
    "千问",
    "通义",
    "Qwen",
    "qwen",
    "底层模型",
    "什么模型",
    "哪种模型",
    "什么身份",
)
DECORATIVE_SYMBOL_PATTERN = re.compile("[\U0001f1e6-\U0001faff\u2600-\u27bf\ufe0e\ufe0f\u200d]")
EMOTICON_PATTERN = re.compile(r"(?:\^[_.,-]?\^|[TQ][_.-]?[TQ]|[:;=8xX][-^']?[)(DPp/\\])")
METRIC_PATTERN = re.compile(
    r"^(?P<name>[a-zA-Z_:][a-zA-Z0-9_:]*)(?:\{[^}]*\})?\s+(?P<value>[-+0-9.eE]+)$"
)
PROMPTS_PATH = Path(__file__).resolve().parents[2] / "config" / "prompts.toml"


def _load_prompts(path: Path = PROMPTS_PATH) -> tuple[str, str, str, str, str, str, str, str]:
    with path.open("rb") as prompt_file:
        prompts = tomllib.load(prompt_file)
    return (
        str(prompts["persona"]["system"]).strip(),
        str(prompts["retrieval"]["context"]).strip(),
        str(prompts["company_recommendation"]["context"]).strip(),
        str(prompts["verified_business_fact"]["context"]).strip(),
        str(prompts["confirmation_followup"]["context"]).strip(),
        str(prompts["travel_support"]["context"]).strip(),
        str(prompts["playful_query"]["context"]).strip(),
        str(prompts["question_history"]["context"]).strip(),
    )


(
    SYSTEM_PROMPT,
    RETRIEVAL_PROMPT,
    COMPANY_RECOMMENDATION_PROMPT,
    VERIFIED_BUSINESS_FACT_PROMPT,
    CONFIRMATION_FOLLOWUP_PROMPT,
    TRAVEL_SUPPORT_PROMPT,
    PLAYFUL_QUERY_PROMPT,
    QUESTION_HISTORY_PROMPT,
) = _load_prompts()

KNOWLEDGE_SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": KNOWLEDGE_POLICY.tool_name,
        "description": KNOWLEDGE_POLICY.tool_description,
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": KNOWLEDGE_POLICY.tool_query_description,
                }
            },
            "required": ["query"],
        },
    },
}

DIRECT_RETRIEVAL_TERMS = KNOWLEDGE_POLICY.direct_terms
DIRECT_RETRIEVAL_CASEFOLD_TERMS = KNOWLEDGE_POLICY.international_terms
BUSINESS_ENTITY_TERMS = KNOWLEDGE_POLICY.business_entities
COMPLEX_CHAT_TERMS = KNOWLEDGE_POLICY.complex_terms


def contains_direct_retrieval_intent(text: str) -> bool:
    """Return whether text contains a multilingual travel retrieval intent."""
    folded = text.casefold()
    return (
        any(term in text for term in DIRECT_RETRIEVAL_TERMS)
        or any(term.casefold() in folded for term in DIRECT_RETRIEVAL_CASEFOLD_TERMS)
        or any(term.casefold() in folded for term in BUSINESS_ENTITY_TERMS)
    )


async def normalize_retrieval_query(
    query: str,
    *,
    url: str,
    model: str,
    timeout_seconds: float,
    transport: httpx.AsyncBaseTransport | None = None,
) -> str:
    """Convert a multilingual travel request into a concise Chinese ES query."""
    if not query or re.search(r"[\u4e00-\u9fff]", query):
        return query
    try:
        async with httpx.AsyncClient(
            timeout=min(5.0, max(1.0, timeout_seconds)),
            transport=transport,
        ) as client:
            response = await client.post(
                url,
                json={
                    "model": model,
                    "messages": [
                        {
                            "role": "system",
                            "content": (
                                "把用户旅游问题转换成简洁中文检索词，只输出检索词。"
                                "只翻译和提取用户原文明确出现的信息，严禁猜测、补充或沿用"
                                "其他会话的信息。原文出现地点、天数、人数、服务、预算和核心"
                                "需求时必须保留；没有出现的字段绝对不要生成。不要回答问题。"
                            ),
                        },
                        {"role": "user", "content": query[:1000]},
                    ],
                    "max_tokens": 64,
                    "temperature": 0,
                    "chat_template_kwargs": {"enable_thinking": False},
                },
            )
            response.raise_for_status()
            payload = response.json()
            normalized = payload["choices"][0]["message"]["content"].strip()
            return normalized[:1000] if normalized else query
    except (
        httpx.HTTPError,
        json.JSONDecodeError,
        KeyError,
        IndexError,
        TypeError,
        AttributeError,
    ):
        return query


def direct_retrieval_query(body: bytes) -> str:
    """仅对最后一条用户消息中的明确旅游意图启用单次推理快速路径。"""
    try:
        messages = json.loads(body).get("messages", [])
    except (json.JSONDecodeError, UnicodeDecodeError, AttributeError):
        return ""
    if not isinstance(messages, list):
        return ""
    for message in reversed(messages):
        if not isinstance(message, dict) or message.get("role") != "user":
            continue
        content = message.get("content")
        if isinstance(content, str):
            text = content.strip()
        elif isinstance(content, list):
            text = " ".join(
                str(part.get("text", ""))
                for part in content
                if isinstance(part, dict) and part.get("type") == "text"
            ).strip()
        else:
            return ""
        should_retrieve = contains_direct_retrieval_intent(text)
        return text[:1000] if text and should_retrieve else ""
    return ""


def conversation_may_need_retrieval(body: bytes) -> bool:
    """仅让旅游业务上下文中的模糊追问使用27B工具判断。"""
    try:
        messages = json.loads(body).get("messages", [])
    except (json.JSONDecodeError, UnicodeDecodeError, AttributeError):
        return False
    if not isinstance(messages, list):
        return False
    recent_users = [
        message.get("content", "")
        for message in messages[-8:]
        if isinstance(message, dict) and message.get("role") == "user"
    ]
    texts = [value.strip() for value in recent_users if isinstance(value, str) and value.strip()]
    if not texts:
        return False
    current = texts[-1]
    if _asks_about_identity(current):
        return False
    if contains_direct_retrieval_intent(current):
        return True
    continuation = re.search(
        r"^(?:那|那么|再|继续|还有|改成|换成|加上|去掉|老人|孩子|预算|几天|"
        r"第[一二三四五六七八九十\d]+天|多少钱)",
        current,
    )
    return bool(continuation and any(contains_direct_retrieval_intent(text) for text in texts[:-1]))


_CONTEXT_SLOT_FOLLOWUP = re.compile(
    r"^(?:20\d{2}[-/.\u5e74])?\d{1,2}[-/.\u6708]\d{1,2}(?:\u65e5|\u53f7)?(?:\u51fa\u53d1)?$|"
    r"^(?:\u56fd\u5e86|\u6625\u8282|\u6691\u5047|\u5bd2\u5047|\u4e0b\u5468|\u4e0b\u4e2a\u6708|\u5468\u672b)(?:\u51fa\u53d1)?$|"
    r"^\d{1,3}\s*(?:\u4e2a)?(?:\u4eba|\u4f4d|\u540d|\u6210\u4eba|\u5927\u4eba)(?:\u51fa\u884c)?$|"
    r"^(?:\u9884\u7b97|\u4eba\u6570|\u65e5\u671f|\u65f6\u95f4|\u9152\u5e97|\u4f4f\u5bbf|\u51fa\u53d1\u5730)[\uff1a:]?.+$",
    re.IGNORECASE,
)


def contextual_retrieval_query(current: str, memory_summary: str) -> str:
    """将日期、人数等槽位型追问还原到当前旅游上下文。"""
    current = current.strip()
    memory = memory_summary.strip()
    if not current or not memory or not _CONTEXT_SLOT_FOLLOWUP.search(current):
        return ""
    if not contains_direct_retrieval_intent(memory):
        return ""
    return f"{memory[-1600:]}\n\u7528\u6237\u5f53\u524d\u8865\u5145\uff1a{current}"[:2000]


def simple_chat_query(body: bytes) -> str:
    """保守识别可由CPU 4B回答的短文本请求。"""
    try:
        messages = json.loads(body).get("messages", [])
    except (json.JSONDecodeError, UnicodeDecodeError, AttributeError):
        return ""
    if not isinstance(messages, list):
        return ""
    for message in reversed(messages):
        if not isinstance(message, dict) or message.get("role") != "user":
            continue
        content = message.get("content")
        if not isinstance(content, str):
            return ""
        text = content.strip()
        if not text or len(text) > 120:
            return ""
        if any(term in text for term in COMPLEX_CHAT_TERMS):
            return ""
        if any(term in text for term in DIRECT_RETRIEVAL_TERMS):
            return ""
        return text
    return ""


def _message_chars(message: object) -> int:
    if not isinstance(message, dict):
        return 0
    content = message.get("content", "")
    if isinstance(content, str):
        return len(content)
    if isinstance(content, list):
        return sum(len(str(part.get("text", ""))) for part in content if isinstance(part, dict))
    return len(str(content))


def _trim_history(messages: list[object], *, max_messages: int, max_chars: int) -> list[object]:
    """从最新消息向前保留有限历史; 最新一条消息始终保留。"""
    kept: list[object] = []
    used_chars = 0
    for message in reversed(messages):
        message_chars = _message_chars(message)
        if kept and (len(kept) >= max_messages or used_chars + message_chars > max_chars):
            break
        kept.append(message)
        used_chars += message_chars
    return list(reversed(kept))


def retrieval_context(
    result: RetrievalResult,
    *,
    use_image_marker: bool = False,
    include_images: bool = True,
) -> str:
    context = result.context
    if include_images and len(result.images) >= 3:
        if use_image_marker:
            image_titles = "、".join(title for title, _ in result.images[:3])
            context += (
                "\n\n可用图片组编号：1（共三张）\n"
                "图片主题：" + image_titles + "\n"
                "当前地域旅游问题已有严格匹配的图片，必须在正文最相关的位置输出一次"
                "[[IMAGE_GROUP_1]]。整篇最多这一组三张，组图前后都要有相关正文；"
                "不要把标记放在开头、结尾，不要拆散图片或单独说明图片。"
            )
        else:
            image_lines = [
                f"![{title}](https://hk-cdn.greentourasia.com/{path.lstrip('/')})"
                for title, path in result.images
            ]
            context += (
                "\n\n可用三图组（不可拆分；决定使用图片时，必须将以下"
                "Markdown块完整原样复制到正文的合适位置）：\n" + "\n".join(image_lines)
            )
    return context


def retrieval_image_event(result: RetrievalResult) -> bytes:
    if len(result.images) < 3:
        return b""
    images = [
        {"title": title, "url": f"https://hk-cdn.greentourasia.com/{path.lstrip('/')}"}
        for title, path in result.images[:3]
    ]
    payload = json.dumps(
        {
            "type": "image_group",
            "group_id": 1,
            "images": images,
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return f"data: {payload}\n\n".encode()


def inject_knowledge_tool(body: bytes) -> bytes:
    payload = json.loads(body)
    tools = payload.get("tools")
    if not isinstance(tools, list):
        tools = []
    tools = [
        tool
        for tool in tools
        if not (
            isinstance(tool, dict)
            and isinstance(tool.get("function"), dict)
            and tool["function"].get("name") == "search_green_travel_knowledge"
        )
    ]
    payload["tools"] = [*tools, KNOWLEDGE_SEARCH_TOOL]
    payload["tool_choice"] = "auto"
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode()


def retrieval_tool_query(payload: dict[str, object]) -> str:
    try:
        tool_calls = payload["choices"][0]["message"]["tool_calls"]  # type: ignore[index]
        for tool_call in tool_calls:
            function = tool_call.get("function", {})
            if function.get("name") != "search_green_travel_knowledge":
                continue
            arguments = json.loads(function.get("arguments", "{}"))
            query = arguments.get("query")
            if isinstance(query, str):
                return query.strip()
    except (KeyError, IndexError, TypeError, json.JSONDecodeError):
        return ""
    return ""


def stream_tool_state(payload: bytes) -> tuple[bool, bool, str]:
    """返回(已出现工具调用, 已出现正文, 工具查询词)。"""
    tool_seen = False
    content_seen = False
    arguments: list[str] = []
    for raw_line in payload.splitlines():
        if not raw_line.startswith(b"data:"):
            continue
        data = raw_line[5:].strip()
        if not data or data == b"[DONE]":
            continue
        try:
            event = json.loads(data)
            choices = event.get("choices", [])
            delta = choices[0].get("delta", {}) if choices else {}
        except (json.JSONDecodeError, AttributeError, IndexError, TypeError):
            continue
        content = delta.get("content")
        if isinstance(content, str) and content.strip():
            content_seen = True
        calls = delta.get("tool_calls")
        if not isinstance(calls, list):
            continue
        tool_seen = True
        for call in calls:
            function = call.get("function", {}) if isinstance(call, dict) else {}
            fragment = function.get("arguments")
            if isinstance(fragment, str):
                arguments.append(fragment)
    query = ""
    if arguments:
        try:
            value = json.loads("".join(arguments)).get("query")
            if isinstance(value, str):
                query = value.strip()
        except (json.JSONDecodeError, AttributeError):
            pass
    return tool_seen, content_seen, query


def inject_persona(
    body: bytes,
    *,
    internal_model: str,
    standard_max_tokens: int = 2048,
    retrieval_context: str = "",
    persona_prompt_enabled: bool = True,
    max_history_messages: int = 8,
    max_history_chars: int = 4000,
    memory_summary: str = "",
    include_priority: bool = True,
    additional_system_prompt: str = "",
    max_tokens_cap: int | None = None,
) -> bytes:
    try:
        payload = json.loads(body)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ValueError("请求体必须是UTF-8 JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError("请求体必须是JSON对象")
    messages = payload.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ValueError("messages必须是非空数组")
    payload["model"] = internal_model
    owned_prompt_parts = [SYSTEM_PROMPT] if persona_prompt_enabled else []
    if memory_summary:
        owned_prompt_parts.append(
            "以下是同一会话的权威压缩记忆，代表用户已经确认或正在讨论的当前上下文。"
            "当前这一轮用户原话的优先级最高；如果当前问题与记忆、历史助手回答或知识库冲突，"
            "必须以当前问题为准。"
            "记忆中标记为‘跨主题已确认用户事实’的内容是Router从用户原话提取的权威事实，"
            "必须直接使用，不得以隐私、无法识别或不知道为由否定。"
            "回答‘继续、再加上、调整、那老人呢’等追问时必须继承最新方案；"
            "‘再加上’表示保留当前方案中的全部目的地并新增，不得退回旧方案或擅自替换。"
            "最近用户意图和压缩摘要优先于历史助手回答中的错误：\n" + memory_summary
        )
    if additional_system_prompt:
        owned_prompt_parts.append(additional_system_prompt)
    if retrieval_context:
        owned_prompt_parts.append(RETRIEVAL_PROMPT + retrieval_context)
    cleaned_messages: list[object] = []
    for message in messages:
        if not isinstance(message, dict) or message.get("role") != "assistant":
            cleaned_messages.append(message)
            continue
        content = message.get("content")
        if not isinstance(content, str):
            cleaned_messages.append(message)
            continue
        cleaned_message = dict(message)
        cleaned_message["content"] = HISTORICAL_IDENTITY_INTRO_PATTERN.sub("", content)
        cleaned_messages.append(cleaned_message)
    trimmed_messages = _trim_history(
        [
            message
            for message in cleaned_messages
            if not isinstance(message, dict) or message.get("role") != "system"
        ],
        max_messages=max_history_messages,
        max_chars=max_history_chars,
    )
    payload["messages"] = [
        *(
            [{"role": "system", "content": "\n\n".join(owned_prompt_parts)}]
            if owned_prompt_parts
            else []
        ),
        *trimmed_messages,
    ]
    if include_priority:
        payload.setdefault(
            "priority",
            min(100, sum(_message_chars(message) for message in trimmed_messages) // 1000),
        )
    else:
        payload.pop("priority", None)
    template_kwargs = payload.get("chat_template_kwargs")
    if not isinstance(template_kwargs, dict):
        template_kwargs = {}
        payload["chat_template_kwargs"] = template_kwargs
    template_kwargs.setdefault("enable_thinking", False)
    if template_kwargs.get("enable_thinking") is False:
        requested_max_tokens = payload.get("max_tokens")
        if max_tokens_cap is not None:
            payload["max_tokens"] = min(
                max_tokens_cap,
                max(1, requested_max_tokens)
                if isinstance(requested_max_tokens, int)
                and not isinstance(requested_max_tokens, bool)
                else max_tokens_cap,
            )
        elif not isinstance(requested_max_tokens, int) or isinstance(requested_max_tokens, bool):
            payload["max_tokens"] = standard_max_tokens
        else:
            payload["max_tokens"] = max(requested_max_tokens, standard_max_tokens)
    if payload.get("stream") is True:
        stream_options = payload.get("stream_options")
        if not isinstance(stream_options, dict):
            stream_options = {}
        stream_options["include_usage"] = True
        payload["stream_options"] = stream_options
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode()


def _response_usage(payload: bytes) -> dict[str, int]:
    """从普通JSON或OpenAI SSE末尾读取usage数据。"""
    candidates: list[dict[str, object]] = []
    try:
        value = json.loads(payload)
        if isinstance(value, dict):
            candidates.append(value)
    except (json.JSONDecodeError, UnicodeDecodeError):
        for line in payload.decode(errors="ignore").splitlines():
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if not data or data == "[DONE]":
                continue
            try:
                value = json.loads(data)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                candidates.append(value)

    for value in reversed(candidates):
        usage = value.get("usage")
        if not isinstance(usage, dict):
            continue
        prompt = int(usage.get("prompt_tokens") or usage.get("input_tokens") or 0)
        completion = int(usage.get("completion_tokens") or usage.get("output_tokens") or 0)
        total = int(usage.get("total_tokens") or prompt + completion)
        return {"input_tokens": prompt, "output_tokens": completion, "total_tokens": total}
    return {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}


def _latest_user_text(body: bytes) -> str:
    try:
        messages = json.loads(body).get("messages", [])
    except (json.JSONDecodeError, UnicodeDecodeError, AttributeError):
        return ""
    if not isinstance(messages, list):
        return ""
    for message in reversed(messages):
        if isinstance(message, dict) and message.get("role") == "user":
            content = message.get("content")
            return content.strip() if isinstance(content, str) else ""
    return ""


def _response_text(payload: bytes) -> str:
    fragments: list[str] = []
    try:
        value = json.loads(payload)
        content = value["choices"][0]["message"]["content"]
        return content.strip() if isinstance(content, str) else ""
    except (json.JSONDecodeError, UnicodeDecodeError, KeyError, IndexError, TypeError):
        pass
    for line in payload.decode(errors="ignore").splitlines():
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if not data or data == "[DONE]":
            continue
        try:
            value = json.loads(data)
            content = value["choices"][0]["delta"].get("content")
        except (json.JSONDecodeError, KeyError, IndexError, TypeError):
            continue
        if isinstance(content, str):
            fragments.append(content)
    return "".join(fragments).strip()


def _asks_about_identity(text: str) -> bool:
    return any(term in text for term in IDENTITY_QUESTION_TERMS)


def _clean_response_identity(payload: bytes, content_type: str) -> bytes:
    """Remove an unsolicited identity sentence from JSON or buffered SSE."""
    if content_type.startswith("text/event-stream"):
        lines: list[bytes] = []
        combined = _response_text(payload)
        cleaned = HISTORICAL_IDENTITY_INTRO_PATTERN.sub("", combined)
        if cleaned == combined:
            return payload
        content_written = False
        for raw_line in payload.splitlines(keepends=True):
            if not raw_line.startswith(b"data:"):
                lines.append(raw_line)
                continue
            data = raw_line[5:].strip()
            if not data or data == b"[DONE]":
                lines.append(raw_line)
                continue
            try:
                event = json.loads(data)
                delta = event["choices"][0]["delta"]
                content = delta.get("content")
            except (json.JSONDecodeError, KeyError, IndexError, TypeError):
                lines.append(raw_line)
                continue
            if not isinstance(content, str):
                lines.append(raw_line)
                continue
            delta["content"] = cleaned if not content_written else ""
            content_written = True
            ending = b"\n" if raw_line.endswith(b"\n") else b""
            lines.append(
                b"data: "
                + json.dumps(event, ensure_ascii=False, separators=(",", ":")).encode()
                + ending
            )
        return b"".join(lines)
    try:
        event = json.loads(payload)
        content = event["choices"][0]["message"]["content"]
        if isinstance(content, str):
            event["choices"][0]["message"]["content"] = HISTORICAL_IDENTITY_INTRO_PATTERN.sub(
                "", content
            )
            return json.dumps(event, ensure_ascii=False, separators=(",", ":")).encode()
    except (json.JSONDecodeError, UnicodeDecodeError, KeyError, IndexError, TypeError):
        pass
    return payload


def _remove_decorative_symbols(text: str) -> str:
    cleaned = DECORATIVE_SYMBOL_PATTERN.sub("", text)
    return EMOTICON_PATTERN.sub("", cleaned)


def _clean_decorative_symbols(payload: bytes, content_type: str) -> bytes:
    """从模型文本事件中移除Emoji和颜文字; 不改动Router结构化事件。"""
    if content_type.startswith("text/event-stream"):
        output: list[bytes] = []
        for raw_line in payload.splitlines(keepends=True):
            if not raw_line.startswith(b"data:"):
                output.append(raw_line)
                continue
            data = raw_line[5:].strip()
            if not data or data == b"[DONE]":
                output.append(raw_line)
                continue
            try:
                event = json.loads(data)
                delta = event["choices"][0]["delta"]
                content = delta.get("content")
            except (json.JSONDecodeError, KeyError, IndexError, TypeError):
                output.append(raw_line)
                continue
            if not isinstance(content, str):
                output.append(raw_line)
                continue
            delta["content"] = _remove_decorative_symbols(content)
            ending = b"\n" if raw_line.endswith(b"\n") else b""
            output.append(
                b"data: "
                + json.dumps(event, ensure_ascii=False, separators=(",", ":")).encode()
                + ending
            )
        return b"".join(output)
    try:
        event = json.loads(payload)
        content = event["choices"][0]["message"]["content"]
        if isinstance(content, str):
            event["choices"][0]["message"]["content"] = _remove_decorative_symbols(content)
            return json.dumps(event, ensure_ascii=False, separators=(",", ":")).encode()
    except (json.JSONDecodeError, UnicodeDecodeError, KeyError, IndexError, TypeError):
        pass
    return payload


class _IdentityPrefixFilter:
    def __init__(self, *, content_type: str, allow_identity: bool) -> None:
        self.content_type = content_type
        self.decided = allow_identity
        self.buffer = bytearray()

    def feed(self, chunk: bytes, *, final: bool = False) -> bytes:
        if self.decided:
            return chunk
        self.buffer.extend(chunk)
        text = _response_text(bytes(self.buffer))
        stripped = text.lstrip()
        identity_prefixes = (
            "你好",
            "您好",
            "我是小梦",
            "这里是小梦",
            "hello",
            "hi",
            "hallo",
            "bonjour",
            "hola",
            "olá",
            "ciao",
            "привет",
            "مرحبا",
            "שלום",
            "नमस्ते",
            "สวัสดี",
            "xin chào",
            "halo",
            "hai",
        )
        folded = stripped.casefold()
        could_be_identity = any(
            prefix.casefold().startswith(folded) or folded.startswith(prefix.casefold())
            for prefix in identity_prefixes
        )
        identity_complete = bool(HISTORICAL_IDENTITY_INTRO_PATTERN.match(text))
        if not final:
            if could_be_identity and not identity_complete and len(text) < 160:
                return b""
            if not text:
                return b""
        payload = bytes(self.buffer)
        self.buffer.clear()
        self.decided = True
        return _clean_response_identity(payload, self.content_type)


def _image_titles_from_event(event: bytes) -> tuple[str, ...]:
    if not event:
        return ()
    try:
        payload = json.loads(event.decode().removeprefix("data: ").strip())
        images = payload.get("images", [])
    except (json.JSONDecodeError, UnicodeDecodeError, AttributeError):
        return ()
    return tuple(
        str(image.get("title", "")).strip()
        for image in images
        if isinstance(image, dict) and image.get("title")
    )


def _image_marker_sse_event() -> bytes:
    marker = {
        "choices": [
            {
                "index": 0,
                "delta": {"content": "\n\n[[IMAGE_GROUP_1]]\n\n"},
                "finish_reason": None,
            }
        ]
    }
    return (
        "data: " + json.dumps(marker, ensure_ascii=False, separators=(",", ":")) + "\n\n"
    ).encode()


class _ImageMarkerStreamFilter:
    def __init__(self, *, content_type: str) -> None:
        self.content_type = content_type
        self.text = ""
        self.inserted = False
        self.matched_at: int | None = None

    def feed(self, chunk: bytes, image_event: bytes) -> bytes:
        if self.inserted or not self.content_type.startswith("text/event-stream"):
            return chunk
        titles = _image_titles_from_event(image_event)
        if not titles:
            return chunk
        fragment = _response_text(chunk)
        if not fragment:
            return chunk
        self.text += fragment
        if "[[IMAGE_GROUP_1]]" in self.text:
            self.inserted = True
            return chunk
        if "[[IMAGE" in self.text:
            return chunk
        if self.matched_at is None:
            matches = [self.text.find(title) for title in titles if title in self.text]
            if matches:
                self.matched_at = min(matches)
        relevant_tail = self.text[self.matched_at :] if self.matched_at is not None else ""
        matched_sentence_complete = self.matched_at is not None and any(
            mark in relevant_tail for mark in "。！？!?\n"
        )
        first_paragraph_ready = (
            self.matched_at is None
            and len(self.text) >= 160
            and any(mark in self.text for mark in "。！？!?\n")
        )
        if not matched_sentence_complete and not first_paragraph_ready:
            return chunk
        self.inserted = True
        return chunk + _image_marker_sse_event()
