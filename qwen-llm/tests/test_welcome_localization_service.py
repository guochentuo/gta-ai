# ruff: noqa: RUF001
from __future__ import annotations

import json

import httpx
import pytest
from router.services.welcome_localization_service import (
    WelcomeLocalizationConfig,
    WelcomeLocalizationService,
    normalize_welcome_locale,
)
from router.services.welcome_template_service import WelcomeTemplate


class _MemoryCache:
    def __init__(self) -> None:
        self.values: dict[str, str] = {}

    def get(self, key: str) -> str | None:
        return self.values.get(key)

    def setex(self, key: str, _: int, value: str) -> bool:
        self.values[key] = value
        return True


def _template() -> WelcomeTemplate:
    return WelcomeTemplate(
        template_id="welcome-default-zh-cn",
        title="您好，我是 GreenTourAI",
        content=(
            "欢迎咨询。\n\n[[IMAGE_GROUP_1]]\n\n"
            "[查看评价](https://share.google/example)\n"
            "[[GTA_PLATFORM_REVIEWS]]\n"
            "[关于我们](https://www.greentourasia.com/about)"
        ),
        images=({"title": "服务", "path": "image/service.jpg"},),
        suggested_questions=("一家四人需要多少钱？",),
        ui_text={"composer_placeholder": "直接问 GreenTourAI"},
        version=21,
    )


def test_normalize_welcome_locale() -> None:
    assert normalize_welcome_locale("zh-CN", "CN") == "zh-CN"
    assert normalize_welcome_locale("zh-Hant", "HK") == "zh-HK"
    assert normalize_welcome_locale("de_DE", "DE") == "de-DE"
    assert normalize_welcome_locale("", "US") == "zh-CN"


@pytest.mark.asyncio
async def test_localizes_once_then_uses_redis_cache() -> None:
    calls = 0
    translated = {
        "title": "Hello, I am GreenTour AI",
        "content": (
            "Welcome.\n\n[[IMAGE_GROUP_1]]\n\n"
            "[Reviews](https://share.google/example)\n"
            "[[GTA_PLATFORM_REVIEWS]]\n"
            "[About us](https://www.greentourasia.com/about)"
        ),
        "suggested_questions": ["How much would a trip for four cost?"],
        "ui_text": {"composer_placeholder": "Ask GreenTourAI"},
    }

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        body = json.loads(request.content)
        assert body["chat_template_kwargs"]["enable_thinking"] is False
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": json.dumps(translated)}}]},
        )

    cache = _MemoryCache()
    service = WelcomeLocalizationService(
        WelcomeLocalizationConfig(redis_cluster_nodes=()),
        transport=httpx.MockTransport(handler),
        redis_client=cache,
    )

    first = await service.localize(_template(), locale_hint="en-US", country_hint="US")
    second = await service.localize(_template(), locale_hint="en-US", country_hint="US")

    assert first.title == "Hello, I am GreenTourAI"
    assert first.content == translated["content"]
    assert second.content == translated["content"]
    assert first.images == _template().images
    assert calls == 1


@pytest.mark.asyncio
async def test_chinese_source_does_not_call_model() -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        raise AssertionError("中文母版不应调用本地化模型")

    service = WelcomeLocalizationService(
        WelcomeLocalizationConfig(),
        transport=httpx.MockTransport(handler),
        redis_client=_MemoryCache(),
    )
    source = _template()
    assert await service.localize(source, locale_hint="zh-CN", country_hint="CN") == source
