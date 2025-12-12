"""
Pydantic schemas for the Wellbeing Data Foundation.

These schemas define the data contracts that must not be changed.
"""

from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime
from enum import Enum


# =============================================================================
# Enums
# =============================================================================


class EntityType(str, Enum):
    """Types of entities in the wellbeing framework."""

    PILLAR = "pillar"
    CORE_VALUE = "core_value"
    SUB_VALUE = "sub_value"
    # Document-level / framework-level chunks (intro, glossary, methodology)
    DOCUMENT = "document"


class ChunkType(str, Enum):
    """Types of chunks for embedding."""

    DEFINITION = "definition"
    EVIDENCE = "evidence"
    COMMENTARY = "commentary"


class EvidenceType(str, Enum):
    """Types of evidence references."""

    QURAN = "quran"
    HADITH = "hadith"
    BOOK = "book"


class Confidence(str, Enum):
    """Confidence levels for answers."""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class Difficulty(str, Enum):
    """Difficulty levels for questions."""

    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class EdgeCreatedMethod(str, Enum):
    """Methods for creating edges in the graph."""

    RULE_EXACT_MATCH = "rule_exact_match"
    RULE_LEMMA = "rule_lemma"
    EMBEDDING_CANDIDATE = "embedding_candidate"
    HUMAN_APPROVED = "human_approved"


class EdgeStatus(str, Enum):
    """Status of edges in the graph."""

    CANDIDATE = "candidate"
    APPROVED = "approved"
    REJECTED = "rejected"


class ParseStatus(str, Enum):
    """Status of reference parsing."""

    SUCCESS = "success"
    FAILED = "failed"
    NEEDS_REVIEW = "needs_review"


# =============================================================================
# Evidence Packet Schema (DO NOT CHANGE)
# =============================================================================


class EvidenceRef(BaseModel):
    """Reference within an evidence packet."""

    type: EvidenceType
    ref: str


class EvidencePacket(BaseModel):
    """
    Evidence packet returned by retrieval.

    This is an authoritative data contract - DO NOT CHANGE.
    """

    chunk_id: str
    entity_type: EntityType
    entity_id: str
    chunk_type: ChunkType
    text_ar: str
    source_doc_id: str
    source_anchor: str
    refs: list[EvidenceRef] = Field(default_factory=list)


# =============================================================================
# Final Response Schema (DO NOT CHANGE)
# =============================================================================


class Purpose(BaseModel):
    """Purpose output from Muḥāsibī middleware."""

    ultimate_goal_ar: str
    constraints_ar: list[str]


class Citation(BaseModel):
    """Citation in the final response."""

    chunk_id: str
    source_anchor: str
    ref: Optional[str] = None


class EntityRef(BaseModel):
    """Entity reference in the final response."""

    type: EntityType
    id: str
    name_ar: str


class FinalResponse(BaseModel):
    """
    Final response schema for the Ask API.

    This is an authoritative data contract - DO NOT CHANGE.
    """

    listen_summary_ar: str
    purpose: Purpose
    path_plan_ar: list[str]
    answer_ar: str
    citations: list[Citation]
    entities: list[EntityRef]
    difficulty: Difficulty
    not_found: bool
    confidence: Confidence


# =============================================================================
# Internal Schemas (can be modified as needed)
# =============================================================================


class SourceAnchor(BaseModel):
    """
    Stable citation anchor scheme.

    Prefer w14:paraId from DOCX XML.
    Fallback: para_index + sha1(normalized_paragraph_text).
    """

    source_doc_id: str  # Immutable per file version, derived from file hash
    anchor_type: str  # docx_para | pdf_page
    anchor_id: str  # w14:paraId or para_index + hash
    anchor_range: Optional[str] = None  # For multi-paragraph chunks


class QuranRef(BaseModel):
    """Parsed Quran reference."""

    surah_name_ar: str
    surah_number: Optional[int] = None
    ayah_number: int
    ref_raw: str  # Original string as appears in document
    ref_norm: str  # Canonical form for indexing
    text_ar: str
    parse_status: ParseStatus = ParseStatus.SUCCESS


class HadithRef(BaseModel):
    """Parsed Hadith reference."""

    collection: str  # e.g., الترمذي
    number: int
    ref_raw: str
    ref_norm: str
    text_ar: str
    parse_status: ParseStatus = ParseStatus.SUCCESS


class IngestionRun(BaseModel):
    """Ingestion run metadata."""

    run_id: str
    source_file_name: str
    source_file_hash: str
    framework_version: str
    status: str
    created_at: datetime
    completed_at: Optional[datetime] = None
    entities_extracted: int = 0
    evidence_extracted: int = 0
    validation_errors: list[str] = Field(default_factory=list)


class Pillar(BaseModel):
    """A pillar in the wellbeing framework."""

    id: str
    name_ar: str
    name_en: Optional[str] = None  # Nullable per MVP policy
    description_ar: Optional[str] = None
    source_anchor: SourceAnchor


class CoreValue(BaseModel):
    """A core value (قيمة كلية / أم)."""

    id: str
    pillar_id: str
    name_ar: str
    name_en: Optional[str] = None
    definition_ar: Optional[str] = None
    source_anchor: SourceAnchor


class SubValue(BaseModel):
    """A sub-value (قيمة جزئية / حفيد)."""

    id: str
    core_value_id: str
    name_ar: str
    name_en: Optional[str] = None
    definition_ar: Optional[str] = None
    source_anchor: SourceAnchor


class Edge(BaseModel):
    """An edge in the knowledge graph."""

    id: str
    from_type: EntityType
    from_id: str
    rel_type: str  # CONTAINS | SUPPORTED_BY | RELATES_TO | CROSS_REFERENCES
    to_type: EntityType
    to_id: str
    created_method: EdgeCreatedMethod
    created_by: str
    justification: Optional[str] = None
    status: EdgeStatus = EdgeStatus.CANDIDATE
    score: Optional[float] = None
    created_at: datetime


class Chunk(BaseModel):
    """A chunk for embedding and retrieval."""

    chunk_id: str
    entity_type: EntityType
    entity_id: str
    chunk_type: ChunkType
    text_ar: str
    text_en: Optional[str] = None
    source_anchor: str
    source_doc_id: str
    token_count_estimate: int
    created_at: datetime

