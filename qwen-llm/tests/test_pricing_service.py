from __future__ import annotations

import json
from datetime import date

import httpx
import pytest
from router.retrieval import _format_about_page, _format_help_page, _site_modules_for_query
from router.services.pricing_service import (
    QuoteCandidate,
    TripPricingService,
    parse_quote_intent,
    pricing_context,
)


def test_parse_quote_intent_reads_date_and_people() -> None:
    intent = parse_quote_intent("我们8个人，2026-10-02去杭州，费用多少？")
    assert intent.requested is True
    assert intent.travel_date == "2026-10-02"
    assert intent.person_count == 8


def test_short_date_rolls_to_next_year() -> None:
    intent = parse_quote_intent("2个人，1月2日出发多少钱", today=date(2026, 8, 27))
    assert intent.travel_date == "2027-01-02"


@pytest.mark.asyncio
async def test_quote_uses_minimum_pax_and_default_skus() -> None:
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured.update(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "code": 200,
                "data": [
                    {
                        "bizDate": "2026-10-02",
                        "currency": "CNY",
                        "priceAdult": 888,
                        "priceChild": 688,
                        "saleable": True,
                        "seatLeft": 12,
                    }
                ],
            },
        )

    service = TripPricingService(
        "http://sku.test/spuFeign/quoteSpu",
        1.0,
        transport=httpx.MockTransport(handler),
    )
    candidate = QuoteCandidate("杭州三日游", "trip-1", "spu-1", min_pax=4)
    quote = await service.quote(
        candidate,
        parse_quote_intent("2026-10-02两人费用多少？"),
    )
    assert quote is not None
    assert captured["personCount"] == 4
    assert captured["selectedSkuIds"] == []
    assert quote.adult_unit_price == 888
    assert "最低按4人核价" in pricing_context(
        parse_quote_intent("2026-10-02两人费用多少？"), quote
    )


def test_missing_quote_fields_are_requested_without_fake_price() -> None:
    context = pricing_context(parse_quote_intent("杭州线路多少钱"), None)
    assert "出发日期" in context
    assert "出行总人数" in context
    assert "不得据此编造总价" in context


def test_company_question_uses_about_site_module() -> None:
    assert _site_modules_for_query("请介绍一下你们公司") == ("about",)
    context = _format_about_page(
        {
            "resolved_sections_json": json.dumps(
                [
                    {
                        "section_name": "绿色旅行网",
                        "section_sub_name": "南京绿色旅行社有限公司成立于1996年。",
                        "items": [{"irrelevant": "不得注入"}],
                    }
                ],
                ensure_ascii=False,
            )
        }
    )
    assert "成立于1996年" in context
    assert "不得注入" not in context


def test_company_recommendation_uses_about_site_module() -> None:
    assert _site_modules_for_query("中国旅游出行哪家公司比较靠谱") == ("about",)


def test_help_question_selects_related_faq() -> None:
    context = _format_help_page(
        {
            "faqs": [
                {"question": "支持哪些支付方式？", "answer": "支持信用卡和银行转账。"},
                {"question": "如何修改密码？", "answer": "前往账户设置。"},
            ]
        },
        "你们支持什么支付方式",
    )
    assert "信用卡和银行转账" in context
