from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import Field, model_validator

from gta_ai.schemas.base import StrictModel


class MetricFact(StrictModel):
    name: str = Field(min_length=1)
    value: int | float
    unit: str = Field(min_length=1)
    source_id: str = Field(min_length=1)
    comparison_value: int | float | None = None


class Evidence(StrictModel):
    evidence_id: str = Field(min_length=1)
    source_type: Literal["metric", "article", "video", "ad", "log", "manual"]
    source_id: str = Field(min_length=1)
    excerpt: str = Field(min_length=1)
    observed_at: datetime | None = None


class Hypothesis(StrictModel):
    statement: str = Field(min_length=1)
    supporting_evidence_ids: list[str] = Field(default_factory=list)
    counter_evidence_ids: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0, le=1)


class RecommendedAction(StrictModel):
    action: str = Field(min_length=1)
    rationale: str = Field(min_length=1)
    expected_effect: str = Field(min_length=1)
    risk: Literal["low", "medium", "high"]
    requires_human_approval: bool = True


class DecisionPacket(StrictModel):
    report_id: str = Field(min_length=1)
    period_start: datetime
    period_end: datetime
    summary: str = Field(min_length=1)
    metrics: list[MetricFact] = Field(default_factory=list)
    evidence: list[Evidence] = Field(default_factory=list)
    hypotheses: list[Hypothesis] = Field(default_factory=list)
    recommended_actions: list[RecommendedAction] = Field(default_factory=list)
    unknowns: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0, le=1)

    @model_validator(mode="after")
    def validate_period_and_evidence_references(self) -> DecisionPacket:
        if self.period_end < self.period_start:
            raise ValueError("period_end must be greater than or equal to period_start")

        known_ids = {item.evidence_id for item in self.evidence}
        referenced_ids = {
            evidence_id
            for hypothesis in self.hypotheses
            for evidence_id in (
                *hypothesis.supporting_evidence_ids,
                *hypothesis.counter_evidence_ids,
            )
        }
        missing = sorted(referenced_ids - known_ids)
        if missing:
            raise ValueError(f"unknown evidence references: {', '.join(missing)}")
        return self


class FinalDecision(StrictModel):
    executive_summary: str = Field(min_length=1)
    accepted_findings: list[str]
    rejected_findings: list[str]
    actions: list[RecommendedAction]
    final_content: str | None
    caveats: list[str]
    confidence: float = Field(ge=0, le=1)
