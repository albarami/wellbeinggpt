"""Pydantic schemas for World Model system dynamics graphs.

Design principles:
1. No duplicate ontology: mechanism_node uses ref_kind + ref_id to anchor to existing tables
2. Polarity as signed integer: +1 or -1, product determines loop type
3. Evidence-based confidence: derived from span count and diversity
4. Framework-bound: all nodes must reference existing framework entities
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, Field


class RefKind(str, Enum):
    """Kind of entity referenced by a mechanism node.
    
    Anchors mechanism nodes to the existing ontology:
    - pillar, core_value, sub_value: existing framework entities
    - mechanism: abstract mechanism (e.g., "تفعيل القوى")
    - outcome: desired outcome (e.g., "الازدهار")
    """
    PILLAR = "pillar"
    CORE_VALUE = "core_value"
    SUB_VALUE = "sub_value"
    MECHANISM = "mechanism"
    OUTCOME = "outcome"


class RelationType(str, Enum):
    """Semantic relation types for mechanism edges.
    
    Each has a default polarity:
    - Positive (+1): ENABLES, REINFORCES, COMPLEMENTS, RESOLVES_WITH, CONDITIONAL_ON
    - Negative (-1): INHIBITS, TENSION_WITH
    """
    ENABLES = "ENABLES"
    REINFORCES = "REINFORCES"
    COMPLEMENTS = "COMPLEMENTS"
    CONDITIONAL_ON = "CONDITIONAL_ON"
    INHIBITS = "INHIBITS"
    TENSION_WITH = "TENSION_WITH"
    RESOLVES_WITH = "RESOLVES_WITH"


# Default polarity by relation type (can be overridden when evidence indicates otherwise)
RELATION_POLARITY_DEFAULTS: dict[str, int] = {
    "ENABLES": 1,       # A increases B
    "REINFORCES": 1,    # A strengthens B
    "COMPLEMENTS": 1,   # A supports B
    "RESOLVES_WITH": 1, # A resolves tension in B
    "CONDITIONAL_ON": 1, # A requires B (positive dependency)
    "INHIBITS": -1,     # A decreases B
    "TENSION_WITH": -1, # A conflicts with B
}


class MechanismNodeBase(BaseModel):
    """Base schema for mechanism node (thin wrapper over existing hierarchy)."""
    
    ref_kind: str = Field(
        ...,
        description="Kind of entity: pillar|core_value|sub_value|mechanism|outcome"
    )
    ref_id: str = Field(
        ...,
        description="Actual ID from referenced table (P001, CV001, SV001, etc.)"
    )
    label_ar: str = Field(
        ...,
        description="Cached display label (derived from referenced entity)"
    )
    source_id: str | None = Field(
        default=None,
        description="Source document UUID"
    )


class MechanismNode(MechanismNodeBase):
    """Full mechanism node with database ID."""
    
    id: str = Field(..., description="UUID of the mechanism node")


class MechanismEdgeSpanBase(BaseModel):
    """Base schema for edge justification span."""
    
    chunk_id: str = Field(..., description="Chunk ID containing the evidence")
    span_start: int = Field(..., ge=0, description="Start offset in chunk text")
    span_end: int = Field(..., ge=0, description="End offset in chunk text")
    quote: str = Field(..., min_length=1, description="Quoted text from chunk")


class MechanismEdgeSpan(MechanismEdgeSpanBase):
    """Full edge span with database ID."""
    
    id: str = Field(..., description="UUID of the span")
    edge_id: str = Field(..., description="UUID of the parent edge")


class MechanismEdgeBase(BaseModel):
    """Base schema for mechanism edge with signed polarity."""
    
    from_node: str = Field(..., description="UUID of source mechanism node")
    to_node: str = Field(..., description="UUID of target mechanism node")
    relation_type: str = Field(
        ...,
        description="Semantic relation: ENABLES|REINFORCES|CONDITIONAL_ON|INHIBITS|TENSION_WITH|RESOLVES_WITH|COMPLEMENTS"
    )
    polarity: Literal[-1, 1] = Field(
        default=1,
        description="Signed polarity: +1 (positive/enabling) or -1 (negative/inhibiting)"
    )
    confidence: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Evidence-based confidence (span count + diversity)"
    )


class MechanismEdge(MechanismEdgeBase):
    """Full mechanism edge with database ID and spans."""
    
    id: str = Field(..., description="UUID of the mechanism edge")
    spans: list[MechanismEdgeSpan] = Field(
        default_factory=list,
        description="Justification spans (hard gate: must have ≥1)"
    )


class FeedbackLoopBase(BaseModel):
    """Base schema for feedback loop."""
    
    edge_ids: list[str] = Field(
        ...,
        min_length=2,
        description="Ordered list of edge UUIDs forming the loop"
    )
    loop_type: Literal["reinforcing", "balancing"] = Field(
        ...,
        description="Loop type derived from polarity product"
    )


class FeedbackLoop(FeedbackLoopBase):
    """Full feedback loop with database ID."""
    
    loop_id: str = Field(..., description="UUID of the loop")


# =============================================================================
# Dataclasses for reasoning (not DB-backed)
# =============================================================================

@dataclass
class DetectedLoop:
    """A detected causal loop with evidence.
    
    Note: loop_summary_ar is generated on-the-fly from edge spans,
    not stored in the database (per adjustment #5).
    """
    
    loop_id: str
    loop_type: str  # "reinforcing" or "balancing"
    edge_ids: list[str]
    nodes: list[str]  # Ordered node labels (ref_kind:ref_id format)
    node_labels_ar: list[str]  # Display labels
    polarities: list[int]  # +1 or -1 per edge
    evidence_spans: list[dict[str, Any]]  # Per-edge evidence from mechanism_edge_span
    
    def generate_summary_ar(self) -> str:
        """Generate summary on-the-fly from edge spans.
        
        If spans exist: concatenate key quotes with connectors.
        If no spans: return "غير منصوص عليه في الإطار".
        """
        if not self.evidence_spans:
            return "غير منصوص عليه في الإطار"
        
        quotes = []
        for span_group in self.evidence_spans[:3]:
            if isinstance(span_group, dict) and span_group.get("quote"):
                q = str(span_group["quote"]).strip()
                if len(q) > 100:
                    q = q[:100] + "..."
                quotes.append(q)
            elif isinstance(span_group, list):
                for sp in span_group[:1]:
                    if isinstance(sp, dict) and sp.get("quote"):
                        q = str(sp["quote"]).strip()
                        if len(q) > 100:
                            q = q[:100] + "..."
                        quotes.append(q)
        
        if not quotes:
            return "غير منصوص عليه في الإطار"
        
        return " ← ".join(quotes)


@dataclass
class InterventionStep:
    """A single step in an intervention plan.
    
    Must reference an actual framework node (adjustment #4).
    """
    
    target_node_ref_kind: str  # core_value|sub_value|mechanism|outcome
    target_node_ref_id: str    # Actual ID from framework
    target_node_label_ar: str  # Display label
    mechanism_reason_ar: str   # Evidence-bound quote
    mechanism_citations: list[dict[str, Any]] = field(default_factory=list)  # chunk_id + span
    expected_impacts: list[str] = field(default_factory=list)  # Downstream node labels
    impact_citations: list[dict[str, Any]] = field(default_factory=list)  # Evidence for impacts


@dataclass
class InterventionPlan:
    """Evidence-bound intervention plan.
    
    Per adjustment #4:
    - Steps must map to existing framework nodes
    - Leading indicators explicitly marked with source
    - No medical/diagnostic claims
    """
    
    goal_ar: str
    steps: list[InterventionStep] = field(default_factory=list)  # 3-7 steps
    # {"indicator_ar": str, "source": "framework"|"غير منصوص"}
    leading_indicators: list[dict[str, Any]] = field(default_factory=list)
    # {"risk_ar": str, "affected_pillar": str, "evidence": list}
    risk_of_imbalance: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class SimulationResult:
    """Result of counterfactual propagation simulation.
    
    Weights are derived from evidence density, not learned parameters.
    Label explicitly states this is approximate.
    """
    
    initial_state: dict[str, float] = field(default_factory=dict)  # node_id -> normalized value (0-1)
    final_state: dict[str, float] = field(default_factory=dict)
    propagation_steps: list[dict[str, Any]] = field(default_factory=list)  # {"step", "node", "delta", "via_edge"}
    label_ar: str = "محاكاة تقريبية وفق روابط الإطار"


# =============================================================================
# Utility functions
# =============================================================================

def compute_loop_type(polarities: list[int]) -> str:
    """Compute loop type from edge polarities.
    
    Algorithm:
    - product of signs = +1 ⇒ reinforcing (even number of -1s)
    - product of signs = -1 ⇒ balancing (odd number of -1s)
    
    Args:
        polarities: List of edge polarities (+1 or -1)
        
    Returns:
        "reinforcing" or "balancing"
    """
    if not polarities:
        return "reinforcing"  # Default for empty
    
    product = 1
    for p in polarities:
        product *= p
    
    return "reinforcing" if product == 1 else "balancing"


def get_default_polarity(relation_type: str) -> int:
    """Get default polarity for a relation type.
    
    Args:
        relation_type: Semantic relation type
        
    Returns:
        +1 or -1
    """
    return RELATION_POLARITY_DEFAULTS.get(relation_type.upper(), 1)


def compute_edge_confidence(
    span_count: int,
    chunk_diversity: int,
    is_direct_quote: bool = True,
) -> float:
    """Compute evidence-based confidence for an edge.
    
    Per recommendation A: confidence derived from:
    - Number of justification spans (more = higher)
    - Span diversity (different chunks = higher)
    - Direct quote vs multi-span entailment
    
    Args:
        span_count: Number of justification spans
        chunk_diversity: Number of distinct chunks
        is_direct_quote: True if single direct quote, False if multi-span entailment
        
    Returns:
        Confidence score in [0.1, 0.95] range
    """
    # Base from spans: 0.1 + 0.1 per span (up to 0.5)
    base = 0.1 + min(0.4, span_count * 0.1)
    
    # Diversity bonus: 0.05 per distinct chunk (up to 0.2)
    diversity_bonus = min(0.2, chunk_diversity * 0.05)
    
    # Direct quote bonus
    quote_bonus = 0.1 if is_direct_quote else 0.0
    
    # Ensure in valid range
    confidence = min(0.95, base + diversity_bonus + quote_bonus)
    
    return round(confidence, 3)


# Forbidden medical/diagnostic claims (adjustment #4)
FORBIDDEN_CLAIMS = [
    "يعالج",
    "يشفي",
    "دواء",
    "علاج طبي",
    "تشخيص",
    "اضطراب",
    "مرض نفسي",
    "اكتئاب سريري",
    "فصام",
    "ثنائي القطب",
    "وصفة طبية",
]


def validate_no_medical_claims(text_ar: str) -> bool:
    """Check if text contains forbidden medical/diagnostic language.
    
    Args:
        text_ar: Arabic text to validate
        
    Returns:
        True if text is safe (no forbidden claims), False otherwise
    """
    if not text_ar:
        return True
    
    text_lower = text_ar.strip()
    for claim in FORBIDDEN_CLAIMS:
        if claim in text_lower:
            return False
    
    return True
