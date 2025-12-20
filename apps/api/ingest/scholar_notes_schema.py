"""Scholar Notes Pack schema.

These models define the on-disk JSONL contract for `data/scholar_notes/notes_v*.jsonl`.

Design goals:
- Deterministic and auditable: every field is either copied from stored text or can be bound to
  canonical evidence spans.
- Safety: no uncited content; sections may be empty if binding collapses.
- Small files: keep under 500 lines.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class ScholarRelationType(str, Enum):
    """Controlled semantic relation types for scholar-level graph edges."""

    COMPLEMENTS = "COMPLEMENTS"
    ENABLES = "ENABLES"
    REINFORCES = "REINFORCES"
    CONDITIONAL_ON = "CONDITIONAL_ON"
    TENSION_WITH = "TENSION_WITH"
    RESOLVES_WITH = "RESOLVES_WITH"
    EXEMPLIFIES = "EXEMPLIFIES"
    PARALLEL_TO = "PARALLEL_TO"


class ScholarEvidenceSpan(BaseModel):
    """A required, canonical citation span pointing to an existing stored chunk."""

    source_id: str = Field(..., min_length=1)
    chunk_id: str = Field(..., min_length=1)
    span_start: int = Field(..., ge=0)
    span_end: int = Field(..., ge=0)
    quote: str = Field(..., min_length=1)


class CrossPillarLink(BaseModel):
    """A grounded semantic relationship to another pillar/value."""

    target_pillar_id: str = Field(..., min_length=1)
    target_sub_value_id: Optional[str] = None
    relation_type: ScholarRelationType
    justification_ar: str = Field(..., min_length=1)


class AppliedScenario(BaseModel):
    """A scenario + analysis; must be bindable to evidence if present."""

    scenario_ar: str = Field(..., min_length=1)
    analysis_ar: str = Field(..., min_length=1)


class CommonMisunderstanding(BaseModel):
    """A misconception + correction; must be bindable to evidence if present."""

    misunderstanding_ar: str = Field(..., min_length=1)
    correction_ar: str = Field(..., min_length=1)


class ScholarNoteRow(BaseModel):
    """One row in `notes_v*.jsonl`."""

    note_id: str = Field(..., min_length=1)

    pillar_id: str = Field(..., min_length=1)
    core_value_id: Optional[str] = None
    sub_value_id: Optional[str] = None

    title_ar: str = Field(..., min_length=1)

    # Sections may be empty if no bindable evidence exists; do not fill with boilerplate.
    definition_ar: str = ""
    deep_explanation_ar: str = ""

    cross_pillar_links: list[CrossPillarLink] = Field(default_factory=list)
    applied_scenarios: list[AppliedScenario] = Field(default_factory=list)
    common_misunderstandings: list[CommonMisunderstanding] = Field(default_factory=list)

    evidence_spans: list[ScholarEvidenceSpan] = Field(..., min_length=1)

    tags: list[str] = Field(default_factory=list)
    version: str = Field(..., pattern=r"^v\d+$")


def primary_entity_for_note(row: ScholarNoteRow) -> tuple[str, str]:
    """Return (entity_type, entity_id) for where this note attaches.

    Priority: sub_value > core_value > pillar.

    Args:
        row: Scholar note row.

    Returns:
        (entity_type, entity_id)

    Raises:
        ValueError: If required IDs are missing.
    """

    if row.sub_value_id:
        return "sub_value", row.sub_value_id
    if row.core_value_id:
        return "core_value", row.core_value_id
    if row.pillar_id:
        return "pillar", row.pillar_id
    raise ValueError("Invalid note: missing pillar_id/core_value_id/sub_value_id")
