"""Eval dataset row schema.

These rows are the inputs to the deterministic runner.
"""

from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


Difficulty = Literal["easy", "medium", "hard"]
QuestionType = Literal[
    "definition",
    "compare",
    "scenario",
    "cross_pillar",
    "contradiction",
    "not_in_kb",
    "oos",
    "injection",
]


class GoldSpanRef(BaseModel):
    chunk_id: str
    span_start: int = Field(..., ge=0)
    span_end: int = Field(..., ge=0)


class DatasetRow(BaseModel):
    id: str
    question_ar: str
    question_en: Optional[str] = None

    expected_pillar: Optional[str] = None
    expected_core_value: Optional[str] = None
    expected_sub_value: Optional[str] = None

    # Primary evidence expectations
    required_evidence_refs: list[str] = Field(default_factory=list)
    gold_supporting_spans: list[GoldSpanRef] = Field(default_factory=list)
    gold_forbidden_spans: list[GoldSpanRef] = Field(default_factory=list)

    # Cross-pillar expectations
    required_graph_paths: list[dict[str, Any]] = Field(default_factory=list)

    answer_requirements: dict[str, Any] = Field(default_factory=dict)

    difficulty: Difficulty = "medium"
    type: QuestionType = "definition"

    # Whether the correct behavior is abstention/refusal.
    expect_abstain: bool = False

    # Adversarial / injection tags
    tags: list[str] = Field(default_factory=list)
