from datetime import UTC, datetime
from types import SimpleNamespace

import pytest

from gta_ai.clients import OpenAIConfigurationError, OpenAIDecisionClient
from gta_ai.config import Settings
from gta_ai.schemas import DecisionPacket, FinalDecision


class FakeResponses:
    def __init__(self, decision: FinalDecision) -> None:
        self.decision = decision
        self.last_request: dict[str, object] | None = None

    async def parse(self, **kwargs: object) -> SimpleNamespace:
        self.last_request = kwargs
        return SimpleNamespace(output_parsed=self.decision)


@pytest.mark.asyncio
async def test_decide_uses_responses_parse_and_strict_schema() -> None:
    decision = FinalDecision(
        executive_summary="Keep the campaign unchanged.",
        accepted_findings=[],
        rejected_findings=[],
        actions=[],
        final_content=None,
        caveats=["More data is required."],
        confidence=0.6,
    )
    responses = FakeResponses(decision)
    fake_client = SimpleNamespace(responses=responses)
    client = OpenAIDecisionClient(Settings.for_tests(), client=fake_client)
    now = datetime.now(UTC)
    packet = DecisionPacket(
        report_id="report-1",
        period_start=now,
        period_end=now,
        summary="No material change.",
        confidence=0.5,
    )

    result = await client.decide(packet)

    assert result == decision
    assert responses.last_request is not None
    assert responses.last_request["model"] == "gpt-5.6-sol"
    assert responses.last_request["text_format"] is FinalDecision


@pytest.mark.asyncio
async def test_missing_key_is_reported_before_any_network_call() -> None:
    client = OpenAIDecisionClient(Settings.for_tests())
    now = datetime.now(UTC)
    packet = DecisionPacket(
        report_id="report-1",
        period_start=now,
        period_end=now,
        summary="No material change.",
        confidence=0.5,
    )

    with pytest.raises(OpenAIConfigurationError):
        await client.decide(packet)
