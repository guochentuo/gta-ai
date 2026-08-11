from datetime import UTC, datetime, timedelta

import pytest
from pydantic import ValidationError

from gta_ai.schemas import DecisionPacket, Evidence, Hypothesis


def test_decision_packet_rejects_unknown_evidence_reference() -> None:
    now = datetime.now(UTC)

    with pytest.raises(ValidationError, match="unknown evidence references"):
        DecisionPacket(
            report_id="report-1",
            period_start=now,
            period_end=now + timedelta(days=1),
            summary="Conversion changed.",
            evidence=[],
            hypotheses=[
                Hypothesis(
                    statement="Traffic quality changed.",
                    supporting_evidence_ids=["missing"],
                    confidence=0.5,
                )
            ],
            confidence=0.5,
        )


def test_decision_packet_accepts_known_evidence_reference() -> None:
    now = datetime.now(UTC)
    packet = DecisionPacket(
        report_id="report-1",
        period_start=now,
        period_end=now,
        summary="Conversion changed.",
        evidence=[
            Evidence(
                evidence_id="evidence-1",
                source_type="metric",
                source_id="metric-1",
                excerpt="CVR increased by 10%.",
            )
        ],
        hypotheses=[
            Hypothesis(
                statement="Traffic quality improved.",
                supporting_evidence_ids=["evidence-1"],
                confidence=0.7,
            )
        ],
        confidence=0.7,
    )

    assert packet.hypotheses[0].supporting_evidence_ids == ["evidence-1"]


def test_strict_schema_rejects_extra_fields() -> None:
    with pytest.raises(ValidationError):
        Evidence(
            evidence_id="evidence-1",
            source_type="manual",
            source_id="manual-1",
            excerpt="Approved by owner.",
            unexpected="not allowed",
        )
