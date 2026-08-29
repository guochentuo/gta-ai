from __future__ import annotations

import asyncio
import fcntl
import json
from pathlib import Path

import httpx
import pytest
from router.app import (
    LANGUAGE_MATCH_PROMPT,
    RETRIEVAL_PROMPT,
    SYSTEM_PROMPT,
    LeaseAcquireRequest,
    PersistentGpuAdmission,
    PriorityWorkloadGate,
    RouterConfig,
    _asks_about_identity,
    _clean_decorative_symbols,
    _clean_response_identity,
    _IdentityPrefixFilter,
    _ImageMarkerStreamFilter,
    _response_text,
    contact_request_event,
    contextual_retrieval_query,
    conversation_may_need_retrieval,
    create_router_app,
    direct_retrieval_query,
    generate_handoff_offer,
    handoff_offer_event,
    inject_knowledge_tool,
    inject_persona,
    language_style_prompt,
    query_explicitly_requests_contact,
    retrieval_context,
    retrieval_image_event,
    retrieval_tool_query,
    simple_chat_query,
    stream_tool_state,
)
from router.retrieval import (
    ElasticsearchRetriever,
    RetrievalConfig,
    RetrievalResult,
    _answer_image_theme,
    _format_trip_product,
    _image_matches_query,
    _is_landscape,
    _product_intent_score,
    _query_needs_images,
)
from router.services.context_classification_service import (
    business_fact_evidence_found,
    classify_context,
    extract_profile_facts,
    is_confirmation_followup,
    is_global_summary_query,
    is_playful_or_impossible_travel_query,
    is_travel_support_query,
    is_user_question_history_query,
    requires_verified_business_fact,
    topics_compatible,
)
from router.summarization import (
    SUMMARY_SYSTEM_PROMPT,
    ConversationSummarizer,
    SummarizationConfig,
    _remove_model_generated_profile_claims,
    estimate_tokens,
)


class _MockStream(httpx.AsyncByteStream):
    def __init__(self, content: bytes) -> None:
        self.content = content

    async def __aiter__(self):
        yield self.content


class _MemoryRedisLock:
    def __init__(self) -> None:
        self._owned = False

    def acquire(self, *, blocking: bool = True) -> bool:
        self._owned = True
        return True

    def release(self) -> None:
        self._owned = False

    def owned(self) -> bool:
        return self._owned


class _MemoryRedisPipeline:
    def __init__(self, client: _MemoryRedis) -> None:
        self.client = client
        self.commands: list[tuple[str, tuple, dict]] = []
        self.in_transaction = False

    def __enter__(self) -> _MemoryRedisPipeline:
        return self

    def __exit__(self, *args) -> None:
        return None

    def watch(self, key: str) -> None:
        return None

    def hget(self, key: str, field: str):
        return self.client.hget(key, field)

    def multi(self) -> None:
        self.in_transaction = True

    def hset(self, *args, **kwargs) -> None:
        self.commands.append(("hset", args, kwargs))

    def expire(self, *args, **kwargs) -> None:
        self.commands.append(("expire", args, kwargs))

    def execute(self) -> list[object]:
        results = [
            getattr(self.client, name)(*args, **kwargs) for name, args, kwargs in self.commands
        ]
        self.commands.clear()
        return results


class _MemoryRedis:
    def __init__(self) -> None:
        self.values: dict[str, dict[str, str]] = {}

    def hmget(self, key: str, *fields: str) -> list[str | None]:
        return [self.hget(key, field) for field in fields]

    def hget(self, key: str, field: str) -> str | None:
        return self.values.get(key, {}).get(field)

    def hset(self, key: str, *, mapping: dict[str, str]) -> int:
        self.values.setdefault(key, {}).update(mapping)
        return len(mapping)

    def hsetnx(self, key: str, field: str, value: str) -> int:
        values = self.values.setdefault(key, {})
        if field in values:
            return 0
        values[field] = value
        return 1

    def expire(self, key: str, seconds: int) -> bool:
        return key in self.values

    def pipeline(self, *, transaction: bool = True) -> _MemoryRedisPipeline:
        return _MemoryRedisPipeline(self)

    def lock(self, *args, **kwargs) -> _MemoryRedisLock:
        return _MemoryRedisLock()


def test_standard_mode_uses_at_least_2048_output_tokens() -> None:
    forwarded = json.loads(
        inject_persona(
            json.dumps(
                {
                    "model": "external",
                    "messages": [{"role": "user", "content": "你好"}],
                    "max_tokens": 512,
                    "chat_template_kwargs": {"enable_thinking": False},
                }
            ).encode(),
            internal_model="internal",
            standard_max_tokens=2048,
        )
    )

    assert forwarded["max_tokens"] == 2048


def test_explicit_contact_query_is_detected() -> None:
    assert query_explicitly_requests_contact("包车顾问怎么联系") is True
    assert query_explicitly_requests_contact("杭州三日游") is False


def test_contact_request_event_contains_no_copy_or_accounts() -> None:
    payload = json.loads(contact_request_event().decode().removeprefix("data: "))

    assert payload == {"type": "contact_request"}


def test_handoff_offer_wraps_27b_copy_without_contact_options() -> None:
    payload = json.loads(
        handoff_offer_event(
            {"prompt": "需要旅行顾问继续帮你安排吗", "action_label": "联系旅行顾问"}
        ).decode().removeprefix("data: ")
    )

    assert payload["type"] == "handoff_offer"
    assert payload["prompt"].startswith("需要旅行顾问继续帮你安排吗")
    assert "contacts" not in payload


@pytest.mark.asyncio
async def test_27b_generates_handoff_offer_in_current_language() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert body["chat_template_kwargs"]["enable_thinking"] is False
        assert "de-DE" in body["messages"][1]["content"]
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": '{"prompt":"Möchten Sie weitere Hilfe?",'
                            '"action_label":"Reiseberater kontaktieren"}'
                        }
                    }
                ]
            },
        )

    result = await generate_handoff_offer(
        "Wie plane ich meine Reise?",
        locale_hint="de-DE",
        country_hint="DE",
        url="http://model/v1/chat/completions",
        model="27b",
        timeout_seconds=5,
        transport=httpx.MockTransport(handler),
    )

    assert result["action_label"] == "Reiseberater kontaktieren"


def test_persona_forbids_repeated_identity_introduction() -> None:
    assert "禁止使用Emoji、颜文字" in SYSTEM_PROMPT
    assert "仅在用户询问身份、开发方或所属公司时" in SYSTEM_PROMPT
    assert "南京绿色旅行社有限公司" in SYSTEM_PROMPT
    assert "亚洲绿色旅游服务有限公司" in SYSTEM_PROMPT
    assert "不得机械重复完整身份句" in SYSTEM_PROMPT
    assert "明确回答没有关系" in SYSTEM_PROMPT
    assert "不得披露、推测或讨论底层模型" in SYSTEM_PROMPT
    assert "不得把自己自称为通义千问" in SYSTEM_PROMPT
    assert "普通问题直接回答" in SYSTEM_PROMPT
    assert "用Markdown粗体标出行程名称" in RETRIEVAL_PROMPT
    assert "不得整句、整段或过度加粗" in RETRIEVAL_PROMPT


def test_router_removes_emoji_and_emoticons_from_sse_text() -> None:
    event = {
        "choices": [{"delta": {"content": "\u884c\u7a0b\u5b89\u6392" + chr(0x1F60A) + " ^_^"}}]
    }
    payload = ("data: " + json.dumps(event, ensure_ascii=False) + "\n\n").encode()

    cleaned = _clean_decorative_symbols(payload, "text/event-stream")

    assert chr(0x1F60A) not in cleaned.decode()
    assert "^_^" not in cleaned.decode()
    assert "行程安排" in cleaned.decode()
    assert "回复语言必须跟随当前这一轮用户消息" in LANGUAGE_MATCH_PROMPT
    assert "不得先生成中文再翻译" in LANGUAGE_MATCH_PROMPT
    regional = language_style_prompt("杭州行程", "zh-HK", "HK")
    assert "locale=zh-HK" in regional
    assert "country=HK" in regional
    english_rule = language_style_prompt("hangzhou travel", "zh-CN", "CN")
    assert "entire response in English only" in english_rule
    assert "Do not answer in Chinese" in english_rule
    traditional_rule = language_style_prompt(
        "爲什麽我的繁體字發出去後變成簡體中文?",
        "zh-CN",
        "CN",
    )
    assert "只能使用繁體中文" in traditional_rule


def test_persona_removes_repeated_identity_intro_from_assistant_history() -> None:
    forwarded = json.loads(
        inject_persona(
            json.dumps(
                {
                    "messages": [
                        {"role": "user", "content": "杭州怎么玩"},
                        {
                            "role": "assistant",
                            "content": "你好!我是小梦,绿色旅行网的AI旅行助手。杭州适合安排三天。",
                        },
                        {"role": "user", "content": "那西安呢"},
                    ]
                },
                ensure_ascii=False,
            ).encode(),
            internal_model="internal",
        )
    )

    assert forwarded["messages"][2]["content"] == "杭州适合安排三天。"


def test_router_removes_unsolicited_identity_intro_from_json_response() -> None:
    payload = json.dumps(
        {
            "choices": [
                {
                    "message": {
                        "content": "你好!我是小梦,绿色旅行网的专属AI旅行助手。杭州适合安排三天。"
                    }
                }
            ]
        },
        ensure_ascii=False,
    ).encode()

    cleaned = json.loads(_clean_response_identity(payload, "application/json"))

    assert cleaned["choices"][0]["message"]["content"] == "杭州适合安排三天。"


@pytest.mark.parametrize(
    "question",
    ("你是哪家公司开发的", "你属于哪家公司", "谁研发了你", "你们的运营方是谁"),
)
def test_company_identity_questions_are_not_filtered(question: str) -> None:
    assert _asks_about_identity(question) is True


def test_router_removes_unsolicited_identity_intro_from_stream() -> None:
    stream_filter = _IdentityPrefixFilter(content_type="text/event-stream", allow_identity=False)
    first = (
        b'data: {"choices":[{"delta":{"content":"'
        b'\\u4f60\\u597d\\uff01\\u6211\\u662f\\u5c0f\\u68a6\\uff0c"}}]}\n\n'
    )
    second = (
        b'data: {"choices":[{"delta":{"content":"'
        b"\\u7eff\\u8272\\u65c5\\u884c\\u7f51\\u7684AI"
        b"\\u65c5\\u884c\\u52a9\\u624b\\u3002"
        b"\\u676d\\u5dde\\u9002\\u5408\\u4e09\\u5929\\u3002"
        b'"}}]}\n\n'
    )

    assert stream_filter.feed(first) == b""
    output = stream_filter.feed(second)

    assert "我是小梦" not in _response_text(output)
    assert _response_text(output) == "杭州适合三天。"


def test_router_removes_german_identity_intro_from_stream() -> None:
    stream_filter = _IdentityPrefixFilter(content_type="text/event-stream", allow_identity=False)
    first = b'data: {"choices":[{"delta":{"content":"Hallo! Ich bin Xiao Meng, "}}]}\n\n'
    second = (
        b'data: {"choices":[{"delta":{"content":"Ihr KI-Reiseassistent von '
        b'GreenTourAsia. Hangzhou eignet sich fuer drei Tage."}}]}\n\n'
    )

    assert stream_filter.feed(first) == b""
    output = stream_filter.feed(second)
    text = _response_text(output)

    assert "Xiao Meng" not in text
    assert "Hangzhou" in text


def test_router_inserts_image_group_after_matching_scenic_sentence() -> None:
    image_filter = _ImageMarkerStreamFilter(content_type="text/event-stream")
    image_event = retrieval_image_event(
        RetrievalResult(
            query="杭州旅游",
            context="",
            hit_count=3,
            elapsed_ms=1,
            images=(
                ("西湖", "image/xihu.jpg"),
                ("灵隐寺", "image/lingyin.jpg"),
                ("西溪湿地", "image/xixi.jpg"),
            ),
        )
    )
    first = (
        "data: "
        + json.dumps(
            {"choices": [{"delta": {"content": "第一天先游览西湖"}}]},
            ensure_ascii=False,
        )
        + "\n\n"
    ).encode()
    second = (
        "data: "
        + json.dumps(
            {"choices": [{"delta": {"content": ",沿白堤慢慢散步。"}}]},
            ensure_ascii=False,
        )
        + "\n\n"
    ).encode()

    assert "[[IMAGE_GROUP_1]]" not in _response_text(image_filter.feed(first, image_event))
    output = image_filter.feed(second, image_event)

    assert _response_text(output).endswith("[[IMAGE_GROUP_1]]")
    assert image_filter.feed(first, image_event) == first


def test_normal_chat_disables_thinking_by_default() -> None:
    forwarded = json.loads(
        inject_persona(
            json.dumps(
                {
                    "model": "external",
                    "messages": [{"role": "user", "content": "你好"}],
                    "max_tokens": 512,
                }
            ).encode(),
            internal_model="internal",
            standard_max_tokens=2048,
        )
    )

    assert forwarded["chat_template_kwargs"] == {"enable_thinking": False}
    assert forwarded["max_tokens"] == 2048


def test_persona_prompt_can_be_disabled_after_adapter_activation() -> None:
    forwarded = json.loads(
        inject_persona(
            json.dumps(
                {
                    "messages": [
                        {"role": "system", "content": "调用者提示不得透传"},
                        {"role": "user", "content": "你是谁"},
                    ]
                },
                ensure_ascii=False,
            ).encode(),
            internal_model="internal",
            persona_prompt_enabled=False,
        )
    )

    assert forwarded["messages"] == [{"role": "user", "content": "你是谁"}]


def test_explicit_thinking_mode_is_preserved() -> None:
    forwarded = json.loads(
        inject_persona(
            json.dumps(
                {
                    "model": "external",
                    "messages": [{"role": "user", "content": "请深入分析"}],
                    "max_tokens": 512,
                    "chat_template_kwargs": {"enable_thinking": True},
                }
            ).encode(),
            internal_model="internal",
            standard_max_tokens=2048,
        )
    )

    assert forwarded["chat_template_kwargs"] == {"enable_thinking": True}
    assert forwarded["max_tokens"] == 512


def test_normal_chat_exposes_native_tool_to_27b() -> None:
    body = inject_knowledge_tool(
        json.dumps({"messages": [{"role": "user", "content": "杭州图文攻略"}]}).encode()
    )
    payload = json.loads(body)
    assert payload["tool_choice"] == "auto"
    assert payload["tools"][0]["function"]["name"] == "search_green_travel_knowledge"


@pytest.mark.parametrize(
    "text",
    [
        "杭州旅游",
        "杭州三日攻略",
        "西湖有哪些景点",
        "南京怎么玩",
        "hangzhou travel",
        "Hangzhou three-day itinerary",
        "杭州観光おすすめ",
        "항저우 여행 추천",
        "杭州旅遊攻略",
    ],
)
def test_clear_travel_intent_uses_direct_retrieval(text: str) -> None:
    body = json.dumps({"messages": [{"role": "user", "content": text}]}).encode()

    assert direct_retrieval_query(body) == text


def test_german_locale_keeps_model_output_rule_in_german() -> None:
    prompt = language_style_prompt(
        "Wie viel kostet eine dreitägige Reise nach Hangzhou?",
        "de-DE",
        "DE",
    )
    assert "Deutsch" in prompt


@pytest.mark.parametrize(
    ("text", "locale", "language_name"),
    [
        ("Combien coûte un voyage à Hangzhou ?", "fr-FR", "French"),
        ("¿Cuánto cuesta un viaje a Hangzhou?", "es-ES", "Spanish"),
        ("Quanto custa uma viagem a Hangzhou?", "pt-BR", "Portuguese"),
        ("Сколько стоит поездка в Ханчжоу?", "ru-RU", "Russian"),
        ("杭州への旅行はいくらですか?", "ja-JP", "日本語"),
        ("항저우 여행 비용은 얼마인가요?", "ko-KR", "한국어"),
        ("การเดินทางไปหางโจวราคาเท่าไหร่", "th-TH", "Thai"),
        ("Chi phí du lịch Hàng Châu là bao nhiêu?", "vi-VN", "Vietnamese"),
    ],
)
def test_major_market_languages_localize_model_and_handoff(
    text: str,
    locale: str,
    language_name: str,
) -> None:
    prompt = language_style_prompt(text, locale, locale.split("-")[-1])

    assert language_name in prompt


@pytest.mark.parametrize("text", ["南京绿色帮我介绍一下", "绿色旅行网是什么", "介绍GreenTourAsia"])
def test_business_entity_uses_direct_retrieval(text: str) -> None:
    body = json.dumps({"messages": [{"role": "user", "content": text}]}).encode()

    assert direct_retrieval_query(body) == text


@pytest.mark.parametrize("text", ["你好", "你是谁", "解释一下量子力学", "谢谢"])
def test_normal_chat_does_not_use_direct_retrieval(text: str) -> None:
    body = json.dumps({"messages": [{"role": "user", "content": text}]}).encode()

    assert direct_retrieval_query(body) == ""


def test_only_latest_user_message_controls_direct_retrieval() -> None:
    body = json.dumps(
        {
            "messages": [
                {"role": "user", "content": "杭州旅游攻略"},
                {"role": "assistant", "content": "好的"},
                {"role": "user", "content": "谢谢"},
            ]
        }
    ).encode()

    assert direct_retrieval_query(body) == ""


def test_travel_followup_can_offer_retrieval_tool() -> None:
    body = json.dumps(
        {
            "messages": [
                {"role": "user", "content": "杭州旅游攻略"},
                {"role": "assistant", "content": "可以安排三天。"},
                {"role": "user", "content": "第二天呢"},
            ]
        },
        ensure_ascii=False,
    ).encode()

    assert conversation_may_need_retrieval(body) is True


def test_unrelated_chat_does_not_offer_retrieval_tool() -> None:
    body = json.dumps(
        {"messages": [{"role": "user", "content": "解释一下操作系统"}]},
        ensure_ascii=False,
    ).encode()

    assert conversation_may_need_retrieval(body) is False


def test_date_followup_uses_server_side_travel_memory() -> None:
    memory = (
        "\u7528\u6237\u60f3\u67e5\u8be2\u676d\u5dde\u4e09\u65e5\u6e38\u65c5\u6e38\u62a5\u4ef7\uff0c"
        "8\u4eba\u51fa\u884c\u3002"
    )

    query = contextual_retrieval_query("10\u67088\u53f7", memory)

    assert "\u676d\u5dde\u4e09\u65e5\u6e38" in query
    assert "8\u4eba" in query
    assert "10\u67088\u53f7" in query


@pytest.mark.parametrize(
    "current",
    ["\u4f60\u5728\u54ea\u91cc", "\u8c22\u8c22", "\u6211\u60f3\u6253\u6b7b\u4f60"],
)
def test_non_slot_chat_does_not_retrieve_from_travel_memory(current: str) -> None:
    memory = "\u676d\u5dde\u4e09\u65e5\u6e38\u65c5\u6e38\u62a5\u4ef7"
    assert contextual_retrieval_query(current, memory) == ""


@pytest.mark.parametrize("current", ["你再那里", "我想打死你", "你在哪里", "谢谢"])
def test_non_travel_short_turn_does_not_inherit_images(current: str) -> None:
    body = json.dumps(
        {
            "messages": [
                {"role": "user", "content": "南京旅游攻略"},
                {"role": "assistant", "content": "可以去中山陵。"},
                {"role": "user", "content": current},
            ]
        },
        ensure_ascii=False,
    ).encode()

    assert conversation_may_need_retrieval(body) is False


def test_identity_topic_switch_ignores_previous_travel_context() -> None:
    body = json.dumps(
        {
            "messages": [
                {"role": "user", "content": "杭州8日游"},
                {"role": "assistant", "content": "可以安排西湖和灵隐寺。"},
                {"role": "user", "content": "千问和你是什么关系"},
            ]
        },
        ensure_ascii=False,
    ).encode()

    assert _asks_about_identity("千问和你是什么关系") is True
    assert conversation_may_need_retrieval(body) is False


@pytest.mark.parametrize(
    "text",
    ["你好", "你是谁", "杭州在哪里", "西湖适合玩多久", "飞机为什么能飞"],
)
def test_short_simple_question_uses_cpu_4b(text: str) -> None:
    body = json.dumps(
        {"messages": [{"role": "user", "content": text}]}, ensure_ascii=False
    ).encode()
    assert simple_chat_query(body) == text


@pytest.mark.parametrize(
    "text",
    [
        "西安旅游攻略",
        "帮我规划杭州三天行程",
        "详细比较杭州和苏州",
        "一家三口预算5000元怎么安排",
        "给我一份带图片的景点推荐",
    ],
)
def test_short_but_complex_question_stays_on_27b(text: str) -> None:
    body = json.dumps(
        {"messages": [{"role": "user", "content": text}]}, ensure_ascii=False
    ).encode()
    assert simple_chat_query(body) == ""


@pytest.mark.parametrize(
    "text",
    [
        "我准备去苏杭玩五天",
        "一家人想去杭州",
        "去苏州玩几天",
    ],
)
def test_regional_travel_phrases_use_direct_retrieval(text: str) -> None:
    body = json.dumps(
        {"messages": [{"role": "user", "content": text}]}, ensure_ascii=False
    ).encode()

    assert direct_retrieval_query(body) == text


def test_history_is_limited_and_short_request_gets_higher_priority() -> None:
    messages = []
    for index in range(6):
        messages.extend(
            [
                {"role": "user", "content": f"问题{index}" + "甲" * 900},
                {"role": "assistant", "content": f"回答{index}" + "乙" * 900},
            ]
        )
    forwarded = json.loads(
        inject_persona(
            json.dumps({"messages": messages}, ensure_ascii=False).encode(),
            internal_model="internal",
            max_history_messages=6,
            max_history_chars=2000,
        )
    )

    history = forwarded["messages"][1:]
    assert len(history) <= 6
    assert sum(len(message["content"]) for message in history) <= 2000
    assert history[-1] == messages[-1]
    assert forwarded["priority"] >= 1


def test_persona_injects_persisted_conversation_summary() -> None:
    forwarded = json.loads(
        inject_persona(
            json.dumps(
                {"messages": [{"role": "user", "content": "继续安排"}]},
                ensure_ascii=False,
            ).encode(),
            internal_model="internal",
            memory_summary="用户计划国庆去杭州, 预算5000元, 不去宋城。",
        )
    )
    assert "压缩记忆" in forwarded["messages"][0]["content"]
    assert "当前这一轮用户原话的优先级最高" in forwarded["messages"][0]["content"]
    assert "不去宋城" in forwarded["messages"][0]["content"]


@pytest.mark.asyncio
async def test_cpu_summarizer_updates_and_persists_summary(tmp_path: Path) -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/chat/completions"
        body = json.loads(request.content)
        assert body["temperature"] == 0.1
        assert body["max_tokens"] == 800
        system_prompt = body["messages"][0]["content"]
        assert system_prompt == SUMMARY_SYSTEM_PROMPT
        assert "所有国家、省市区、城市、乡镇、景区、景点" in system_prompt
        assert "老人、儿童、孕妇、残障" in system_prompt
        assert "预算、币种、价格、房型、交通、住宿、餐饮要求" in system_prompt
        assert "再加上" in system_prompt
        assert "表示保留原方案并新增" in system_prompt
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "目的地杭州; 预算5000元; 不去宋城。"}}]},
        )

    summarizer = ConversationSummarizer(
        SummarizationConfig(enabled=True),
        transport=httpx.MockTransport(handler),
        redis_client=_MemoryRedis(),
    )
    summary = await summarizer.update("session-1", "安排杭州", "建议游览西湖")
    assert "预算5000元" in summary
    memory = summarizer.get("session-1")
    assert summary in memory
    assert "安排杭州" in memory
    assert "建议游览西湖" in memory


@pytest.mark.asyncio
async def test_cpu_summarizer_does_not_merge_different_topics() -> None:
    captured_turns: list[list[dict[str, str]]] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        content = json.loads(body["messages"][1]["content"])
        captured_turns.append(content["待合并对话"])
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {"topic": "general", "summary": "投诉主题摘要"},
                                ensure_ascii=False,
                            )
                        }
                    }
                ]
            },
        )

    redis_client = _MemoryRedis()
    summarizer = ConversationSummarizer(
        SummarizationConfig(enabled=True, recent_turns=0),
        transport=httpx.MockTransport(handler),
        redis_client=redis_client,
    )
    summarizer.record_turn("topic-isolation", "杭州旅游怎么玩", "杭州行程内容")
    summarizer.record_turn("topic-isolation", "导游骂我怎么办", "投诉处理内容")

    await summarizer.update("topic-isolation")

    assert len(captured_turns) == 1
    assert captured_turns[0][0]["user"] == "导游骂我怎么办"
    assert "杭州行程内容" in summarizer.get("topic-isolation", current_query="杭州旅游")
    support_memory = summarizer.get("topic-isolation", current_query="投诉导游")
    assert "投诉主题摘要" in support_memory


def test_place_memory_is_independent_from_summary(tmp_path: Path) -> None:
    summarizer = ConversationSummarizer(
        SummarizationConfig(enabled=True), redis_client=_MemoryRedis()
    )

    summarizer.remember_places("places", ("南京", "乌镇", "西湖"))
    summarizer.remember_places("places", ("乌镇", "灵隐寺"))

    memory = summarizer.get("places", current_query="之前说过哪些地方")
    assert "南京、西湖、乌镇、灵隐寺" in memory


def test_contact_text_is_claimed_only_once_per_session(tmp_path: Path) -> None:
    summarizer = ConversationSummarizer(
        SummarizationConfig(enabled=True), redis_client=_MemoryRedis()
    )

    assert summarizer.contact_was_shown("conversion") is False
    assert summarizer.mark_contact_shown("conversion") is True
    assert summarizer.contact_was_shown("conversion") is True
    assert summarizer.mark_contact_shown("conversion") is False


@pytest.mark.asyncio
async def test_pending_turn_is_available_before_async_summary_finishes(
    tmp_path: Path,
) -> None:
    summarizer = ConversationSummarizer(
        SummarizationConfig(enabled=True), redis_client=_MemoryRedis()
    )

    summarizer.record_turn("session-fast", "第二天住哪里", "建议住在西湖附近")

    memory = summarizer.get("session-fast")
    assert "尚未合并进摘要" in memory
    assert "第二天住哪里" in memory
    assert "助手历史提议" in memory
    assert "非用户事实" in memory
    assert "建议住在西湖附近" in memory


def test_confirmed_display_name_is_available_across_topics() -> None:
    summarizer = ConversationSummarizer(
        SummarizationConfig(enabled=True), redis_client=_MemoryRedis()
    )

    summarizer.record_turn("profile", "我是小王", "好的, 小王。")
    summarizer.record_turn(
        "profile",
        "我是谁",
        "由于我无法获取您的个人隐私, 所以不知道您的具体身份。",
    )

    identity_memory = summarizer.get("profile", current_query="我是谁")
    travel_memory = summarizer.get("profile", current_query="杭州旅游攻略")
    assert "跨主题已确认用户事实" in identity_memory
    assert "用户希望被称呼为" in identity_memory
    assert "小王" in identity_memory
    assert "无法获取您的个人隐私" not in identity_memory
    assert "用户希望被称呼为" not in travel_memory
    assert "小王" not in travel_memory


def test_latest_display_name_replaces_previous_name() -> None:
    summarizer = ConversationSummarizer(
        SummarizationConfig(enabled=True), redis_client=_MemoryRedis()
    )

    summarizer.record_turn("profile-latest", "我是小张", "好的。")
    summarizer.record_turn("profile-latest", "叫我小王", "好的。")

    memory = summarizer.get("profile-latest", current_query="我叫什么")
    assert "用户希望被称呼为" in memory
    assert "小王" in memory
    assert "小张" not in memory


def test_old_pending_turn_migrates_confirmed_display_name() -> None:
    redis_client = _MemoryRedis()
    key = "gta:ai:session:{profile-migration}"
    redis_client.hset(
        key,
        mapping={
            "pending": json.dumps(
                [{"user": "我是小王", "assistant": "好的。", "topic": "identity"}],
                ensure_ascii=False,
            ),
            "active_topic": "identity",
        },
    )
    summarizer = ConversationSummarizer(
        SummarizationConfig(enabled=True), redis_client=redis_client
    )

    memory = summarizer.get("profile-migration", current_query="我是谁")

    assert "用户希望被称呼为" in memory
    assert "小王" in memory
    assert "小王" in str(redis_client.hget(key, "profile_facts"))


@pytest.mark.parametrize(
    ("text", "expected"),
    [("我是小王", "小王"), ("我叫Alice", "Alice"), ("call me David", "David")],
)
def test_extract_display_name(text: str, expected: str) -> None:
    assert extract_profile_facts(text) == {"display_name": expected}


def test_memory_context_has_token_limit_and_discards_unrelated_old_turns(
    tmp_path: Path,
) -> None:
    summarizer = ConversationSummarizer(
        SummarizationConfig(
            enabled=True,
            max_context_tokens=180,
            recent_turns=1,
        ),
        redis_client=_MemoryRedis(),
    )
    summarizer.record_turn("bounded", "杭州预算是多少", "预算是5000元")
    for index in range(4):
        summarizer.record_turn(
            "bounded",
            f"无关寒暄{index}",
            "这是没有长期价值的闲聊" + "甲" * 100,
        )
    summarizer.record_turn("bounded", "西湖怎么去", "可以乘坐地铁到西湖")

    memory = summarizer.get("bounded", current_query="继续说杭州预算")

    assert estimate_tokens(memory) <= 180
    assert "预算是5000元" in memory
    assert "西湖怎么去" in memory
    assert "无关寒暄0" not in memory


def test_context_topics_separate_unrelated_conversation_memory() -> None:
    summarizer = ConversationSummarizer(
        SummarizationConfig(enabled=True), redis_client=_MemoryRedis()
    )
    summarizer.record_turn("classified", "杭州8日游", "建议安排西湖和灵隐寺")
    summarizer.record_turn("classified", "千问和你是什么关系", "Qwen是底层模型")

    identity_memory = summarizer.get("classified", current_query="你是什么模型")
    travel_memory = summarizer.get("classified", current_query="继续安排杭州行程")

    assert "Qwen是底层模型" in identity_memory
    assert "西湖和灵隐寺" not in identity_memory
    assert "西湖和灵隐寺" in travel_memory
    assert "Qwen是底层模型" not in travel_memory
    assert classify_context("杭州三日游") == "trip"
    assert classify_context("费用是多少") == "quote"
    assert classify_context("中国旅游出行最好的公司") == "company"
    assert classify_context("推荐一家靠谱的旅行社") == "company"
    assert classify_context("庹国臣是谁", "quote") == "general"
    assert classify_context("导游骂我怎么办") == "support"
    assert is_confirmation_followup("需要") is True
    assert is_global_summary_query("全部总结") is True
    assert is_user_question_history_query("所有我发给你的问题你发给我一下") is True
    assert is_travel_support_query("导游辱骂我怎么办") is True
    assert requires_verified_business_fact("庹国臣是谁") is True
    assert requires_verified_business_fact("这家公司法定代表人是谁") is True
    assert business_fact_evidence_found("庹国臣是谁", "杭州旅游资料") is False
    assert business_fact_evidence_found("庹国臣是谁", "人物: 庹国臣") is True
    assert topics_compatible("trip", "quote") is True
    assert classify_context("中国的月球 美国的火星一日游赶吗") == "smalltalk"
    assert classify_context("哈哈哈哈哈哈") == "smalltalk"
    assert is_playful_or_impossible_travel_query("那就去韩国的水星玩两天") is True
    assert is_playful_or_impossible_travel_query("月球主题乐园怎么玩") is False


def test_question_is_not_saved_as_display_name() -> None:
    assert extract_profile_facts("我是疯了吗") == {}


def test_summary_discards_profile_claim_invented_from_assistant_answer() -> None:
    summary = (
        "用户身份已明确为庹先生。"
        "用户询问中国月球和美国火星的一日游是否赶。"
    )

    cleaned = _remove_model_generated_profile_claims(summary)

    assert "庹先生" not in cleaned
    assert "中国月球" in cleaned


def test_smalltalk_does_not_receive_old_trip_places_or_display_name() -> None:
    summarizer = ConversationSummarizer(
        SummarizationConfig(enabled=True), redis_client=_MemoryRedis()
    )
    summarizer.record_turn("smalltalk-isolated", "我是小王", "好的。")
    summarizer.record_turn("smalltalk-isolated", "杭州旅游", "可以去西湖。")
    summarizer.remember_places("smalltalk-isolated", ("杭州", "西湖"))

    memory = summarizer.get("smalltalk-isolated", current_query="哈哈哈哈")

    assert "小王" not in memory
    assert "杭州" not in memory
    assert "西湖" not in memory


def test_global_summary_combines_all_topics() -> None:
    summarizer = ConversationSummarizer(
        SummarizationConfig(enabled=True), redis_client=_MemoryRedis()
    )
    summarizer.record_turn("global-summary", "杭州怎么玩", "杭州行程内容")
    summarizer.record_turn("global-summary", "导游骂我怎么办", "投诉处理内容")

    memory = summarizer.get("global-summary", current_query="全部总结")

    assert "杭州行程内容" in memory
    assert "投诉处理内容" in memory


def test_user_question_history_preserves_original_questions_across_topics() -> None:
    summarizer = ConversationSummarizer(
        SummarizationConfig(enabled=True), redis_client=_MemoryRedis()
    )
    summarizer.record_turn("question-history", "杭州四天怎么玩", "杭州行程回答")
    summarizer.record_turn("question-history", "乌镇两天够吗", "乌镇行程回答")
    summarizer.record_turn("question-history", "你是谁", "身份回答")

    memory = summarizer.get(
        "question-history",
        current_query="所有我发给你的问题你发给我一下",
    )

    assert "1. 杭州四天怎么玩" in memory
    assert "2. 乌镇两天够吗" in memory
    assert "3. 你是谁" in memory
    assert "杭州行程回答" not in memory


def test_confirmation_followup_inherits_pending_topic_and_offer() -> None:
    summarizer = ConversationSummarizer(
        SummarizationConfig(enabled=True), redis_client=_MemoryRedis()
    )
    summarizer.record_turn(
        "support-confirmation",
        "导游骂我怎么办",
        "我可以帮你整理投诉要点和沟通话术, 需要吗?",
    )

    memory = summarizer.get("support-confirmation", current_query="需要")

    assert "整理投诉要点和沟通话术" in memory


def test_ambiguous_followup_keeps_newest_turn_before_older_turn(tmp_path: Path) -> None:
    summarizer = ConversationSummarizer(
        SummarizationConfig(
            enabled=True,
            max_context_tokens=180,
            recent_turns=2,
        ),
        redis_client=_MemoryRedis(),
    )
    summarizer.record_turn(
        "latest",
        "南京玩三天",
        "南京三天旧方案" + "旧" * 180,
    )
    summarizer.record_turn(
        "latest",
        "乌镇再玩一天",
        "当前方案改为南京两天加乌镇一天",
    )

    memory = summarizer.get("latest", current_query="如何再加上西塘")

    assert "南京两天加乌镇一天" in memory
    assert memory.index("南京两天加乌镇一天") < memory.index("南京三天旧方案")


def test_image_candidates_require_image_intent_and_matching_destination() -> None:
    hangzhou_cover = {
        "title": "杭州三日游攻略",
        "tags": ["浙江", "杭州", "西湖"],
        "cover_url": "image/hangzhou.jpg",
    }

    assert _query_needs_images("杭州旅游攻略") is True
    assert _query_needs_images("我们一家五口想去杭州旅游") is True
    assert _query_needs_images("想去杭州度假") is True
    assert _query_needs_images("杭州旅游需要多少钱") is False
    assert _query_needs_images("为什么南京绿色可以赚钱") is False
    assert _image_matches_query("杭州旅游攻略", hangzhou_cover) is True
    assert _image_matches_query("四川旅游攻略", hangzhou_cover) is False
    assert _query_needs_images("hangzhou travel") is True
    assert (
        _image_matches_query(
            "hangzhou travel",
            {**hangzhou_cover, "city_name": "杭州"},
        )
        is True
    )
    assert _query_needs_images("how much does it cost") is False
    assert _is_landscape({"width": 1600, "height": 900}) is True
    assert _is_landscape({"width": 900, "height": 1600}) is False
    assert _is_landscape({"width": 0, "height": 0}) is False
    assert _answer_image_theme("杭州包车推荐", "杭州有很多景点") == (
        "包车用车",
        ("包车", "用车", "车辆", "车型", "轿车", "商务车", "大巴", "司机", "接送", "接机", "送机"),
    )
    assert _answer_image_theme("最近怎么样", "可以安排包车、酒店和导游服务") is None


def test_trip_product_context_preserves_quote_boundary() -> None:
    product = _format_trip_product(
        {
            "_score": 0.91,
            "_source": {
                "document_kind": "trip",
                "trip_id": "1001",
                "title": "杭州乌镇三日游",
                "summary": "适合家庭慢节奏游览。",
                "duration": "3天2晚",
                "display_url": "/trip/1001",
                "itinerary_days": [
                    {
                        "day_number": 1,
                        "title": "西湖经典游",
                        "from_address": "断桥",
                        "to_address": "雷峰塔",
                    }
                ],
                "packages": [
                    {
                        "package_name": "家庭套餐",
                        "sku_name": "四人家庭套餐",
                        "applicable_people": "2成人2儿童",
                        "min_pax": 4,
                    }
                ],
                "sale_summary": {
                    "reference_start_price": 2999.0,
                    "currency": "CNY",
                    "requires_realtime_quote": True,
                },
            },
        },
        350,
    )

    assert "产品名称: 杭州乌镇三日游" in product
    assert "第1天: 西湖经典游; 断桥 → 雷峰塔" in product
    assert "四人家庭套餐; 适用人群: 2成人2儿童; 至少4人" in product
    assert "默认人数与核价口径: 2成人2儿童; 默认核价人数不得低于4人" in product
    assert "参考起价: 2999 CNY (不是实时报价)" in product
    assert "准确价格需要根据日期、人数和套餐实时询价" in product


def test_trip_product_intent_ranking_prefers_current_destination_and_duration() -> None:
    matching = {
        "_score": 0.8,
        "_source": {
            "title": "杭州乌镇三天家庭游",
            "summary": "适合四人家庭",
            "duration": "3天2晚",
            "tags": ["杭州", "乌镇", "亲子"],
            "packages": [{"applicable_people": "四人家庭"}],
        },
    }
    unrelated = {
        "_score": 0.9,
        "_source": {
            "title": "北京五日文化游",
            "summary": "故宫与长城",
            "duration": "5天4晚",
            "tags": ["北京"],
            "packages": [],
        },
    }

    query = "一家4个人去杭州乌镇玩3天"
    assert _product_intent_score(query, matching) > _product_intent_score(query, unrelated)


@pytest.mark.asyncio
async def test_vehicle_question_uses_vehicle_assets_without_scenic_fallback() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            json={
                "hits": {
                    "hits": [
                        {
                            "_source": {
                                "title": "杭州七座商务车包车服务",
                                "tags": ["包车", "商务车", "司机"],
                                "cover_url": "image/vehicle.jpg",
                                "width": 1600,
                                "height": 900,
                            }
                        },
                        {
                            "_source": {
                                "title": "杭州西湖风光",
                                "tags": ["杭州", "西湖"],
                                "cover_url": "image/xihu.jpg",
                                "width": 1600,
                                "height": 900,
                            }
                        },
                    ]
                }
            },
        )

    retriever = ElasticsearchRetriever(
        RetrievalConfig(enabled=True),
        transport=httpx.MockTransport(handler),
    )
    images = await retriever.retrieve_images_from_answer(
        "杭州包车可以选择五座轿车、七座商务车或者旅游大巴,司机负责接送。" * 2,
        "杭州包车推荐",
    )

    assert images == (("杭州七座商务车包车服务", "image/vehicle.jpg"),)
    assert len(requests) == 1
    assert "gta_scenic_admin" not in requests[0].url.path


def test_stream_retrieval_context_uses_short_image_marker() -> None:
    result = RetrievalResult(
        query="杭州旅游",
        context="杭州资料",
        hit_count=1,
        elapsed_ms=10,
        images=(
            ("西湖", "image/xihu.jpg"),
            ("灵隐寺", "image/lingyin.jpg"),
            ("西溪", "image/xixi.jpg"),
        ),
    )

    context = retrieval_context(result, use_image_marker=True)

    assert "[[IMAGE_GROUP_1]]" in context
    assert "必须在正文最相关的位置输出一次" in context
    assert "整篇最多这一组三张" in context
    assert "https://" not in context
    assert "![" not in context


def test_image_group_event_is_structured_and_limited_to_three_images() -> None:
    result = RetrievalResult(
        query="杭州旅游",
        context="杭州资料",
        hit_count=4,
        elapsed_ms=10,
        images=(
            ("西湖", "image/xihu.jpg"),
            ("灵隐寺", "image/lingyin.jpg"),
            ("西溪", "image/xixi.jpg"),
            ("运河", "image/canal.jpg"),
        ),
    )

    event = retrieval_image_event(result).decode()
    payload = json.loads(event.removeprefix("data: ").strip())

    assert payload["type"] == "image_group"
    assert payload["group_id"] == 1
    assert len(payload["images"]) == 3


def test_less_than_three_images_skip_entire_image_group() -> None:
    result = RetrievalResult(
        query="杭州旅游",
        context="杭州资料",
        hit_count=2,
        elapsed_ms=10,
        images=(("西湖", "image/xihu.jpg"), ("灵隐寺", "image/lingyin.jpg")),
    )

    assert "[[IMAGE_GROUP_1]]" not in retrieval_context(result, use_image_marker=True)
    assert retrieval_image_event(result) == b""


def test_stream_tool_state_assembles_incremental_arguments() -> None:
    payload = b"\n\n".join(
        [
            b'data: {"choices":[{"delta":{"tool_calls":'
            b'[{"function":{"arguments":"{\\"query\\":"}}]}}]}',
            b'data: {"choices":[{"delta":{"tool_calls":'
            b'[{"function":{"arguments":"\\"Hangzhou\\"}"}}]}}]}',
            b"data: [DONE]",
        ]
    )
    assert stream_tool_state(payload) == (True, False, "Hangzhou")


def test_retrieval_query_only_comes_from_27b_tool_call() -> None:
    assert retrieval_tool_query({"choices": [{"message": {"content": "否"}}]}) == ""
    payload = {
        "choices": [
            {
                "message": {
                    "tool_calls": [
                        {
                            "function": {
                                "name": "search_green_travel_knowledge",
                                "arguments": '{"query":"杭州旅游攻略"}',
                            }
                        }
                    ]
                }
            }
        ]
    }
    assert retrieval_tool_query(payload) == "杭州旅游攻略"


@pytest.mark.asyncio
async def test_retriever_embeds_and_searches_business_indices() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.path == "/v1/embeddings":
            return httpx.Response(200, json={"data": [{"embedding": [0.1] * 1024}]})
        request_body = json.loads(request.content)
        if "knn" not in request_body:
            return httpx.Response(
                200,
                json={
                    "hits": {
                        "hits": [
                            {
                                "_score": 8.5,
                                "_source": {
                                    "document_id": "scenic:core",
                                    "document_kind": "scenic",
                                    "title": "无锡梅园核心景区",
                                    "scenic_name": "无锡梅园",
                                    "cover_url": "image/meiyuan-core.jpg",
                                    "width": 1600,
                                    "height": 900,
                                },
                            }
                        ]
                    }
                },
            )
        return httpx.Response(
            200,
            json={
                "hits": {
                    "hits": [
                        {
                            "_score": 0.91,
                            "_source": {
                                "document_id": "scenic:1",
                                "document_kind": "scenic",
                                "title": "无锡梅园",
                                "summary": "适合亲子赏花。",
                                "cover_url": "image/meiyuan.jpg",
                                "width": 1200,
                                "height": 800,
                                "display_url": "scenic/meiyuan",
                            },
                        }
                    ]
                }
            },
        )

    retriever = ElasticsearchRetriever(
        RetrievalConfig(
            enabled=True,
            embedding_url="http://embedding/v1/embeddings",
            elasticsearch_url="https://elasticsearch",
            elasticsearch_username="elastic",
            elasticsearch_password="secret",
            elasticsearch_indices=("gta_scenic_admin",),
        ),
        transport=httpx.MockTransport(handler),
    )
    result = await retriever.retrieve("无锡亲子景点")

    assert result.hit_count == 1
    assert "无锡梅园" in result.context
    assert "image/meiyuan.jpg" not in result.context
    assert result.images[0] == ("无锡梅园", "image/meiyuan-core.jpg")
    assert requests[0].url.path == "/v1/embeddings"
    assert requests[1].url.path == "/gta_scenic_admin/_search"
    assert requests[2].url.path == "/gta_scenic_admin/_search"
    search_body = json.loads(requests[1].content)
    assert len(search_body["knn"]) == 2
    assert len(search_body["knn"][0]["query_vector"]) == 1024
    assert search_body["knn"][1]["filter"]["bool"]["filter"][1] == {
        "term": {"document_kind": "trip"}
    }
    assert search_body["min_score"] == 0.72
    scenic_body = json.loads(requests[2].content)
    assert scenic_body["query"]["bool"]["must"][0]["multi_match"]["fields"][0] == ("scenic_name^5")


def test_persona_includes_owned_retrieval_context() -> None:
    forwarded = json.loads(
        inject_persona(
            json.dumps(
                {"messages": [{"role": "user", "content": "介绍无锡梅园"}]},
                ensure_ascii=False,
            ).encode(),
            internal_model="internal",
            retrieval_context="标题: 无锡梅园\n内容: 适合亲子赏花。",
        )
    )

    assert forwarded["messages"][0]["role"] == "system"
    assert forwarded["messages"][0]["content"].startswith(SYSTEM_PROMPT)
    assert RETRIEVAL_PROMPT in forwarded["messages"][0]["content"]
    assert "无锡梅园" in forwarded["messages"][0]["content"]
    assert "[[IMAGE_GROUP_1]]" not in forwarded["messages"][0]["content"]
    assert "图片检索和插入完全由Router处理" in forwarded["messages"][0]["content"]
    assert "图片地址、图片Markdown、HTML" in forwarded["messages"][0]["content"]
    assert forwarded["messages"][1] == {
        "role": "user",
        "content": "介绍无锡梅园",
    }


@pytest.mark.asyncio
async def test_priority_gate_runs_realtime_before_waiting_history() -> None:
    gate = PriorityWorkloadGate(1)
    await gate.acquire(9)
    order: list[str] = []

    async def wait_for_slot(name: str, priority: int) -> None:
        await gate.acquire(priority)
        order.append(name)

    historical = asyncio.create_task(wait_for_slot("history", 9))
    realtime = asyncio.create_task(wait_for_slot("realtime", 1))
    await asyncio.sleep(0)
    await gate.release()
    await asyncio.wait_for(realtime, timeout=1)
    assert order == ["realtime"]
    await gate.release()
    await asyncio.wait_for(historical, timeout=1)
    assert order == ["realtime", "history"]
    await gate.release()


@pytest.mark.asyncio
async def test_cancelled_priority_waiter_is_removed_immediately() -> None:
    gate = PriorityWorkloadGate(1)
    await gate.acquire(9)
    waiting = asyncio.create_task(gate.acquire(9))
    await asyncio.sleep(0)
    assert (await gate.snapshot())["historical_waiting"] == 1
    waiting.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiting
    assert (await gate.snapshot())["historical_waiting"] == 0
    await gate.release()


@pytest.mark.asyncio
async def test_router_injects_owned_persona(tmp_path: Path) -> None:
    captured: list[bytes] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "GET" and request.url.path == "/v1/models":
            return httpx.Response(200, json={"data": [{"id": "base"}]})
        captured.append(request.content)
        return httpx.Response(
            200,
            headers={"content-type": "application/json"},
            stream=_MockStream(b'{"choices":[{"message":{"content":"ok"}}]}'),
        )

    transport = httpx.MockTransport(handler)
    config = RouterConfig(
        backend_url="http://backend",
        state_path=tmp_path / "state.json",
        gpu_lock_path=tmp_path / "gpu.lock",
        admission_db_path=tmp_path / "admission.sqlite3",
        wake_timeout_seconds=1,
        backend_poll_seconds=0.01,
    )
    app = create_router_app(config, transport=transport)
    payload = {
        "model": "Qwen/Qwen3.8-27B-FP8",
        "messages": [{"role": "user", "content": "你是谁?"}],
    }
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://router"
    ) as client:
        response = await client.post("/v1/chat/completions", json=payload)
        await response.aread()

    assert response.status_code == 200
    assert len(captured) == 1
    forwarded = json.loads(captured[0])
    assert forwarded["model"] == payload["model"]
    assert forwarded["messages"][0]["role"] == "system"
    assert forwarded["messages"][0]["content"].startswith(SYSTEM_PROMPT)
    assert LANGUAGE_MATCH_PROMPT in forwarded["messages"][0]["content"]
    assert forwarded["messages"][1] == {"role": "user", "content": "你是谁?"}


@pytest.mark.asyncio
async def test_empty_messages_returns_vector_welcome_without_calling_model(
    tmp_path: Path,
) -> None:
    requested_paths: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requested_paths.append(request.url.path)
        if request.url.path == "/v1/embeddings":
            return httpx.Response(200, json={"data": [{"embedding": [0.1] * 1024}]})
        if request.url.path.endswith("/_search"):
            return httpx.Response(
                200,
                json={
                    "hits": {
                        "hits": [
                            {
                                "_id": "welcome-default",
                                "_source": {
                                    "template_id": "welcome-default",
                                    "title": "欢迎来到绿色旅行网",
                                    "content": "欢迎内容\n\n[[IMAGE_GROUP_1]]\n\n继续了解行程。",
                                    "images": [
                                        {"title": "杭州1", "path": "image/hangzhou-1.jpg"},
                                        {"title": "杭州2", "path": "image/hangzhou-2.jpg"},
                                        {"title": "杭州3", "path": "image/hangzhou-3.jpg"},
                                    ],
                                "suggested_questions": ["杭州怎么玩？"],  # noqa: RUF001
                                    "version": 1,
                                },
                            }
                        ]
                    }
                },
            )
        raise AssertionError(f"unexpected backend call: {request.url}")

    app = create_router_app(
        RouterConfig(
            backend_url="http://backend",
            state_path=tmp_path / "state.json",
            gpu_lock_path=tmp_path / "gpu.lock",
            admission_db_path=tmp_path / "admission.sqlite3",
            retrieval_embedding_url="http://embedding/v1/embeddings",
            retrieval_elasticsearch_url="http://es",
            welcome_template_character_interval_ms=0,
        ),
        transport=httpx.MockTransport(handler),
    )
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://router"
    ) as client:
        response = await client.post(
            "/v1/chat/completions",
            json={
                "model": "green-travel-ai",
                "keyword": "杭州亲子游",
                "stream": True,
            },
        )
        body = (await response.aread()).decode()

    assert response.status_code == 200
    streamed_content = "".join(
        str(json.loads(line[6:]).get("choices", [{}])[0].get("delta", {}).get("content", ""))
        for line in body.splitlines()
        if line.startswith("data: {") and '"choices"' in line
    )
    assert "欢迎来到绿色旅行网" in streamed_content
    assert '"type":"image_group"' in body
    assert '"type":"handoff_offer"' in body
    assert "/v1/embeddings" in requested_paths
    assert "/v1/chat/completions" not in requested_paths


@pytest.mark.asyncio
async def test_router_replaces_caller_system_identity(tmp_path: Path) -> None:
    captured: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "GET" and request.url.path == "/v1/models":
            return httpx.Response(200, json={"data": [{"id": "base"}]})
        captured.append(request)
        return httpx.Response(200, stream=_MockStream(b"done"))

    app = create_router_app(
        RouterConfig(
            backend_url="http://backend",
            state_path=tmp_path / "state.json",
            gpu_lock_path=tmp_path / "gpu.lock",
            admission_db_path=tmp_path / "admission.sqlite3",
        ),
        transport=httpx.MockTransport(handler),
    )
    payload = {
        "model": "Qwen/Qwen3.8-27B-FP8",
        "messages": [
            {"role": "system", "content": "你是通义千问"},
            {"role": "user", "content": "你是谁?"},
        ],
        "stream": True,
    }
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://router"
    ) as client:
        response = await client.post(
            "/v1/chat/completions",
            json=payload,
            headers={"X-GTA-Persona": "green-travel"},
        )
        response_body = await response.aread()
        assert response_body.startswith(b"done")
        assert b'"type":"handoff_offer"' not in response_body

    assert response.status_code == 200
    assert len(captured) == 1
    forwarded = json.loads(captured[0].content)
    assert forwarded["messages"][0]["role"] == "system"
    assert forwarded["messages"][0]["content"].startswith(SYSTEM_PROMPT)
    assert LANGUAGE_MATCH_PROMPT in forwarded["messages"][0]["content"]
    assert forwarded["messages"][1] == {"role": "user", "content": "你是谁?"}
    assert forwarded["stream"] is True
    assert captured[0].headers["x-gta-persona"] == "green-travel"


@pytest.mark.asyncio
async def test_router_rejects_invalid_chat_json(
    tmp_path: Path,
) -> None:
    captured: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(200, stream=_MockStream(b"done"))

    app = create_router_app(
        RouterConfig(
            backend_url="http://backend",
            state_path=tmp_path / "state.json",
            gpu_lock_path=tmp_path / "gpu.lock",
            admission_db_path=tmp_path / "admission.sqlite3",
        ),
        transport=httpx.MockTransport(handler),
    )
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://router"
    ) as client:
        response = await client.post(
            "/v1/chat/completions",
            content=b"not-json",
            headers={"X-GTA-Persona": "green-travel"},
        )

    assert response.status_code == 422
    assert not captured


@pytest.mark.asyncio
async def test_router_records_workload_lifecycle(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "GET":
            return httpx.Response(200, json={"data": []})
        return httpx.Response(200, stream=_MockStream(b"done"))

    state_path = tmp_path / "state.json"
    app = create_router_app(
        RouterConfig(
            backend_url="http://backend",
            state_path=state_path,
            gpu_lock_path=tmp_path / "gpu.lock",
            admission_db_path=tmp_path / "admission.sqlite3",
        ),
        transport=httpx.MockTransport(handler),
    )
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://router"
    ) as client:
        response = await client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "test"}]},
        )
        assert await response.aread() == b"done"

    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert state["active_requests"] == 0
    assert state["last_workload_at"] > 0


@pytest.mark.asyncio
async def test_request_registers_active_before_waiting_for_training_lock(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "GET":
            return httpx.Response(200, json={"data": []})
        return httpx.Response(200, stream=_MockStream(b"done"))

    state_path = tmp_path / "state.json"
    lock_path = tmp_path / "gpu.lock"
    lock_path.touch()
    training_lock = lock_path.open("a+")
    fcntl.flock(training_lock, fcntl.LOCK_EX)
    app = create_router_app(
        RouterConfig(
            backend_url="http://backend",
            state_path=state_path,
            gpu_lock_path=lock_path,
            admission_db_path=tmp_path / "admission.sqlite3",
        ),
        transport=httpx.MockTransport(handler),
    )
    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://router"
        ) as client:
            request = asyncio.create_task(
                client.post(
                    "/v1/chat/completions",
                    json={"messages": [{"role": "user", "content": "test"}]},
                )
            )
            await asyncio.sleep(0.05)
            assert not request.done()
            state = json.loads(state_path.read_text(encoding="utf-8"))
            assert state["active_requests"] == 1
            fcntl.flock(training_lock, fcntl.LOCK_UN)
            response = await asyncio.wait_for(request, timeout=1)
            assert response.content == b"done"
    finally:
        training_lock.close()


@pytest.mark.asyncio
async def test_explicit_request_cancel_releases_all_router_state(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "GET":
            return httpx.Response(200, json={"data": []})
        return httpx.Response(200, stream=_MockStream(b"done"))

    lock_path = tmp_path / "gpu.lock"
    lock_path.touch()
    training_lock = lock_path.open("a+")
    fcntl.flock(training_lock, fcntl.LOCK_EX)
    app = create_router_app(
        RouterConfig(
            backend_url="http://backend",
            state_path=tmp_path / "state.json",
            gpu_lock_path=lock_path,
            admission_db_path=tmp_path / "admission.sqlite3",
            request_timeout_seconds=5,
        ),
        transport=httpx.MockTransport(handler),
        memory_provider=lambda: (1000, 8000),
    )
    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://router"
        ) as client:
            pending = asyncio.create_task(
                client.post(
                    "/v1/chat/completions",
                    json={"messages": [{"role": "user", "content": "test"}]},
                    headers={"X-GTA-Request-ID": "task:cancel-me"},
                )
            )
            await asyncio.sleep(0)
            for _ in range(50):
                state = (await client.get("/_gta/runtime")).json()
                if state["tracked_requests"] == 1:
                    break
                await asyncio.sleep(0.01)
            duplicate = await client.post(
                "/v1/chat/completions",
                json={"messages": [{"role": "user", "content": "test"}]},
                headers={"X-GTA-Request-ID": "task:cancel-me"},
            )
            assert duplicate.status_code == 409
            cancelled = await client.delete("/_gta/requests/task:cancel-me")
            assert cancelled.json() == {
                "request_id": "task:cancel-me",
                "cancelled": True,
            }
            response = await asyncio.wait_for(pending, timeout=1)
            assert response.status_code == 499
            assert (await client.delete("/_gta/requests/task:cancel-me")).json()[
                "cancelled"
            ] is False
            state = (await client.get("/_gta/runtime")).json()
            admission = (await client.get("/_gta/admission")).json()
            assert state["active_requests"] == 0
            assert state["active_workloads"] == 0
            assert state["tracked_requests"] == 0
            assert state["historical_waiting"] == 0
            assert admission["leases"] == []
            assert admission["waiters"] == []
    finally:
        fcntl.flock(training_lock, fcntl.LOCK_UN)
        training_lock.close()


@pytest.mark.asyncio
async def test_gpu_lease_survives_manager_restart_and_expires(tmp_path: Path) -> None:
    database = tmp_path / "admission.sqlite3"
    manager = PersistentGpuAdmission(
        database,
        poll_seconds=0.01,
        minimum_free_mb=100,
        memory_provider=lambda: (1000, 8000),
    )
    lease = await manager.acquire(
        LeaseAcquireRequest(
            owner="worker:feature:1",
            workload_class="REALTIME_FEATURE",
            requested_memory_mb=2000,
            ttl_seconds=5,
        )
    )
    restarted = PersistentGpuAdmission(
        database,
        poll_seconds=0.01,
        minimum_free_mb=100,
        memory_provider=lambda: (1000, 8000),
    )
    snapshot = await restarted.snapshot()
    assert [item["lease_id"] for item in snapshot["leases"]] == [lease["lease_id"]]
    assert await restarted.release(str(lease["lease_id"]), "worker:feature:1")
    assert (await restarted.snapshot())["leases"] == []


@pytest.mark.asyncio
async def test_realtime_waiter_runs_before_historical_waiter(tmp_path: Path) -> None:
    manager = PersistentGpuAdmission(
        tmp_path / "admission.sqlite3",
        poll_seconds=0.01,
        minimum_free_mb=100,
        memory_provider=lambda: (42000, 4000),
    )
    active = await manager.acquire(
        LeaseAcquireRequest(
            owner="active:understanding",
            workload_class="REALTIME_UNDERSTANDING",
            ttl_seconds=30,
        )
    )
    order: list[str] = []

    async def acquire_named(name: str, workload_class: str) -> dict[str, object]:
        lease = await manager.acquire(
            LeaseAcquireRequest(
                owner=name,
                workload_class=workload_class,
                ttl_seconds=30,
                wait_seconds=2,
            )
        )
        order.append(name)
        return lease

    historical = asyncio.create_task(acquire_named("history", "HISTORICAL_UNDERSTANDING"))
    await asyncio.sleep(0.02)
    realtime = asyncio.create_task(acquire_named("realtime", "REALTIME_UNDERSTANDING"))
    await asyncio.sleep(0.02)
    await manager.release(str(active["lease_id"]), "active:understanding")
    realtime_lease = await asyncio.wait_for(realtime, timeout=1)
    assert order == ["realtime"]
    await manager.release(str(realtime_lease["lease_id"]), "realtime")
    historical_lease = await asyncio.wait_for(historical, timeout=1)
    assert order == ["realtime", "history"]
    await manager.release(str(historical_lease["lease_id"]), "history")


@pytest.mark.asyncio
async def test_admission_http_contract_rejects_unknown_class(tmp_path: Path) -> None:
    app = create_router_app(
        RouterConfig(
            backend_url="http://backend",
            state_path=tmp_path / "state.json",
            gpu_lock_path=tmp_path / "gpu.lock",
            admission_db_path=tmp_path / "admission.sqlite3",
        ),
        transport=httpx.MockTransport(lambda _: httpx.Response(200, json={})),
        memory_provider=lambda: (1000, 8000),
    )
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://router"
    ) as client:
        bad = await client.post(
            "/_gta/admission/leases",
            json={"owner": "worker:1", "workload_class": "UNKNOWN"},
        )
        assert bad.status_code == 422
        good = await client.post(
            "/_gta/admission/leases",
            json={
                "owner": "worker:1",
                "workload_class": "REALTIME_FEATURE",
                "requested_memory_mb": 1000,
            },
        )
        assert good.status_code == 200
        lease = good.json()
        released = await client.delete(
            f"/_gta/admission/leases/{lease['lease_id']}",
            params={"owner": "worker:1"},
        )
        assert released.json() == {"released": True}


@pytest.mark.asyncio
async def test_single_gpu_admission_never_runs_nvenc_whisper_or_27b_together(
    tmp_path: Path,
) -> None:
    manager = PersistentGpuAdmission(
        tmp_path / "admission.sqlite3",
        poll_seconds=0.01,
        minimum_free_mb=2048,
        maximum_active=1,
        memory_provider=lambda: (36591, 8870),
    )
    playback = await manager.acquire(
        LeaseAcquireRequest(
            owner="nvenc:1",
            workload_class="REALTIME_PLAYBACK",
            requested_memory_mb=1024,
            ttl_seconds=30,
        )
    )
    with pytest.raises(TimeoutError):
        await manager.acquire(
            LeaseAcquireRequest(
                owner="whisper:1",
                workload_class="REALTIME_FEATURE",
                requested_memory_mb=6144,
                ttl_seconds=30,
            )
        )
    await manager.release(str(playback["lease_id"]), "nvenc:1")
