"""行程产品实时核价：ES召回产品，gta-sku计算实际价格。"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date
from typing import Any

import httpx

from ..prompt_config import PROMPTS


@dataclass(frozen=True)
class QuoteCandidate:
    title: str
    trip_id: str
    spu_id: str
    min_pax: int = 1
    applicable_people: str = ""
    display_url: str = ""


@dataclass(frozen=True)
class QuoteIntent:
    requested: bool
    travel_date: str = ""
    person_count: int | None = None


@dataclass(frozen=True)
class ExactQuote:
    candidate: QuoteCandidate
    travel_date: str
    requested_people: int
    quoted_people: int
    currency: str
    adult_unit_price: float | None
    child_unit_price: float | None
    young_unit_price: float | None
    service_fee: float | None
    single_supplement: float | None
    seat_left: int | None
    saleable: bool
    message: str


_PRICE_PATTERN = re.compile(
    r"(?:多少钱|费用|价格|报价|预算|价钱|收费|成本|cost|price|quote|budget|how much)",
    re.IGNORECASE,
)
_DATE_PATTERN = re.compile(r"(?P<year>20\d{2})[-/.年](?P<month>\d{1,2})[-/.月](?P<day>\d{1,2})日?")
_SHORT_DATE_PATTERN = re.compile(r"(?<!\d)(?P<month>\d{1,2})月(?P<day>\d{1,2})日?")
_PEOPLE_PATTERN = re.compile(
    r"(?P<count>\d{1,3})\s*(?:个)?(?:人|位|名|成人|大人|游客|旅客|persons?|people|travelers?)",
    re.IGNORECASE,
)
_CN_PEOPLE_PATTERN = re.compile(r"(?P<count>[一二两三四五六七八九十]{1,3})\s*(?:人|位|名)")


def parse_quote_intent(text: str, *, today: date | None = None) -> QuoteIntent:
    person_match = _PEOPLE_PATTERN.search(text)
    person_count = int(person_match.group("count")) if person_match else None
    if person_count is None and (cn_match := _CN_PEOPLE_PATTERN.search(text)):
        person_count = _chinese_number(cn_match.group("count"))
    travel_date = ""
    match = _DATE_PATTERN.search(text)
    try:
        if match:
            travel_date = date(
                int(match.group("year")), int(match.group("month")), int(match.group("day"))
            ).isoformat()
        else:
            short_match = _SHORT_DATE_PATTERN.search(text)
            if short_match:
                current = today or date.today()
                candidate = date(
                    current.year,
                    int(short_match.group("month")),
                    int(short_match.group("day")),
                )
                if candidate < current:
                    candidate = date(current.year + 1, candidate.month, candidate.day)
                travel_date = candidate.isoformat()
    except ValueError:
        travel_date = ""
    return QuoteIntent(
        requested=bool(_PRICE_PATTERN.search(text)),
        travel_date=travel_date,
        person_count=person_count if person_count and person_count > 0 else None,
    )


class TripPricingService:
    def __init__(
        self,
        url: str,
        timeout_seconds: float,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._url = url
        self._timeout_seconds = timeout_seconds
        self._transport = transport

    async def quote(self, candidate: QuoteCandidate, intent: QuoteIntent) -> ExactQuote | None:
        if not intent.travel_date or intent.person_count is None:
            return None
        quoted_people = max(intent.person_count, candidate.min_pax, 1)
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(self._timeout_seconds), transport=self._transport
        ) as client:
            response = await client.post(
                self._url,
                json={
                    "spuId": candidate.spu_id,
                    "selectedSkuIds": [],
                    "startDate": intent.travel_date,
                    "endDate": intent.travel_date,
                    "personCount": quoted_people,
                },
            )
            response.raise_for_status()
            payload = response.json()
        if not isinstance(payload, dict) or int(payload.get("code", 500)) != 200:
            return None
        data = payload.get("data")
        rows = data if isinstance(data, list) else [data] if isinstance(data, dict) else []
        row = next(
            (
                item
                for item in rows
                if isinstance(item, dict) and item.get("bizDate") == intent.travel_date
            ),
            next((item for item in rows if isinstance(item, dict)), None),
        )
        if not isinstance(row, dict):
            return None
        return ExactQuote(
            candidate=candidate,
            travel_date=intent.travel_date,
            requested_people=intent.person_count,
            quoted_people=quoted_people,
            currency=str(row.get("currency") or "CNY"),
            adult_unit_price=_number(row.get("priceAdult")),
            child_unit_price=_number(row.get("priceChild")),
            young_unit_price=_number(row.get("priceYoung")),
            service_fee=_number(row.get("serviceFee")),
            single_supplement=_number(row.get("singleSupplement")),
            seat_left=_integer(row.get("seatLeft")),
            saleable=row.get("saleable") is True,
            message=str(row.get("message") or row.get("unsaleableReason") or ""),
        )


def pricing_context(intent: QuoteIntent, quote: ExactQuote | None) -> str:
    if not intent.requested:
        return ""
    missing = []
    if not intent.travel_date:
        missing.append("出发日期")
    if intent.person_count is None:
        missing.append("出行总人数")
    if missing:
        return PROMPTS.pricing_missing.format(missing="、".join(missing))
    if quote is None:
        return PROMPTS.pricing_unavailable
    fields = [
        "【绿色旅行网实时核价结果】",
        f"匹配线路: {quote.candidate.title}",
        f"产品编号: {quote.candidate.trip_id}",
        f"出发日期: {quote.travel_date}",
        f"用户人数: {quote.requested_people}",
        f"核价人数: {quote.quoted_people}",
        f"币种: {quote.currency}",
        f"成人单价: {_display_number(quote.adult_unit_price)}",
        f"儿童单价: {_display_number(quote.child_unit_price)}",
        f"婴幼儿单价: {_display_number(quote.young_unit_price)}",
        f"服务费: {_display_number(quote.service_fee)}",
        f"单房差: {_display_number(quote.single_supplement)}",
        f"剩余名额: {quote.seat_left if quote.seat_left is not None else '未提供'}",
        f"可售: {'是' if quote.saleable else '否'}",
        f"核价提示: {quote.message or '无'}",
        f"产品页面: {quote.candidate.display_url}",
        PROMPTS.pricing_quote_footer,
    ]
    if quote.quoted_people > quote.requested_people:
        fields.append(
            PROMPTS.pricing_minimum_group.format(
                quoted_people=quote.quoted_people,
                requested_people=quote.requested_people,
            )
        )
    return "\n".join(fields)


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _chinese_number(value: str) -> int | None:
    digits = {
        "一": 1,
        "二": 2,
        "两": 2,
        "三": 3,
        "四": 4,
        "五": 5,
        "六": 6,
        "七": 7,
        "八": 8,
        "九": 9,
    }
    if value == "十":
        return 10
    if "十" in value:
        left, right = value.split("十", 1)
        return (digits.get(left, 1) * 10) + digits.get(right, 0)
    return digits.get(value)


def _integer(value: Any) -> int | None:
    number = _number(value)
    return int(number) if number is not None else None


def _display_number(value: float | None) -> str:
    return "未提供" if value is None else f"{value:g}"
