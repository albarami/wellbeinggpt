"""Rule extractor types (state + extracted dataclasses).

Reason: keep `rule_extractor.py` <500 LOC (project rule).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional


class ExtractorState(Enum):
    """States for the extraction state machine."""

    INITIAL = auto()
    IN_PILLARS_LIST = auto()
    IN_PILLAR = auto()
    IN_CORE_VALUES_LIST = auto()
    IN_CORE_VALUE = auto()
    IN_SUB_VALUES_LIST = auto()
    IN_SUB_VALUE = auto()
    IN_DEFINITION = auto()
    IN_EVIDENCE = auto()


@dataclass
class ExtractedEvidence:
    """An extracted evidence reference."""

    evidence_type: str  # quran | hadith | book
    ref_raw: str
    text_ar: str
    source_doc: str
    source_hash: str
    source_anchor: str
    raw_text: str
    para_index: int
    refs: list[dict] = field(default_factory=list)


@dataclass
class ExtractedDefinition:
    """An extracted definition block."""

    text_ar: str
    source_doc: str
    source_hash: str
    source_anchor: str
    raw_text: str
    para_indices: list[int] = field(default_factory=list)
    refs: list[dict] = field(default_factory=list)


@dataclass
class ExtractedSubValue:
    """An extracted sub-value."""

    id: str
    name_ar: str
    source_doc: str
    source_hash: str
    source_anchor: str
    raw_text: str
    para_index: int
    definition: Optional[ExtractedDefinition] = None
    evidence: list[ExtractedEvidence] = field(default_factory=list)


@dataclass
class ExtractedCoreValue:
    """An extracted core value."""

    id: str
    name_ar: str
    source_doc: str
    source_hash: str
    source_anchor: str
    raw_text: str
    para_index: int
    definition: Optional[ExtractedDefinition] = None
    evidence: list[ExtractedEvidence] = field(default_factory=list)
    sub_values: list[ExtractedSubValue] = field(default_factory=list)


@dataclass
class ExtractedPillar:
    """An extracted pillar."""

    id: str
    name_ar: str
    source_doc: str
    source_hash: str
    source_anchor: str
    raw_text: str
    para_index: int
    description_ar: Optional[str] = None
    core_values: list[ExtractedCoreValue] = field(default_factory=list)


@dataclass
class ExtractionResult:
    """Result of the extraction process."""

    source_doc_id: str
    source_file_hash: str
    source_doc: str
    framework_version: str
    pillars: list[ExtractedPillar] = field(default_factory=list)
    validation_errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    total_core_values: int = 0
    total_sub_values: int = 0
    total_evidence: int = 0

