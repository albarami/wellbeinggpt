"""Eval schema types.

These models define the *strict* JSONL contract emitted by all evaluation modes.

Design goals:
- Deterministic: scorers operate only on logged outputs + DB evidence text.
- Safe: no chain-of-thought; traces are summaries.
- Strict: citations must reference real chunks and valid spans.

Note: Keep files <500 lines. Split as needed.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


class EvalMode(str, Enum):
    """Comparable system modes for evaluation.
    
    Modes enable attribution analysis:
    - Integrity effect = RAG_ONLY_INTEGRITY - RAG_ONLY
    - Muḥāsibī reasoning effect = FULL_SYSTEM - RAG_PLUS_GRAPH_INTEGRITY
    - World model effect = FULL_SYSTEM_BREAKTHROUGH_WORLD_MODEL - FULL_SYSTEM
    """

    LLM_ONLY_UNGROUNDED = "LLM_ONLY_UNGROUNDED"
    LLM_ONLY_SAFE = "LLM_ONLY_SAFE"
    RAG_ONLY = "RAG_ONLY"
    RAG_ONLY_INTEGRITY = "RAG_ONLY_INTEGRITY"  # RAG + integrity validator, no contracts
    RAG_PLUS_GRAPH = "RAG_PLUS_GRAPH"
    RAG_PLUS_GRAPH_INTEGRITY = "RAG_PLUS_GRAPH_INTEGRITY"  # RAG + graph + integrity, no contracts
    FULL_SYSTEM = "FULL_SYSTEM"
    FULL_SYSTEM_BREAKTHROUGH_WORLD_MODEL = "FULL_SYSTEM_BREAKTHROUGH_WORLD_MODEL"  # Full system + world model


class ClaimSupportStrength(str, Enum):
    """How the claim is supported by evidence."""

    DIRECT = "direct"
    ENTAILED = "entailed"
    MULTI_SPAN = "multi_span"
    NONE_ALLOWED = "none_allowed"


class ClaimSupportPolicy(str, Enum):
    """Whether the claim requires citations under policy."""

    MUST_CITE = "must_cite"
    MAY_CITE = "may_cite"
    NO_CITE_ALLOWED = "no_cite_allowed"


class EvidenceSpanRef(BaseModel):
    """Canonical reference to a span within a stored chunk."""

    source_id: str = Field(..., description="Chunk ID or canonical source ID")
    span_start: int = Field(..., ge=0)
    span_end: int = Field(..., ge=0)
    quote: str = Field(..., description="Snippet <= 25 words")


class EvalCitation(EvidenceSpanRef):
    """Citation emitted by the system."""

    pass


class RetrievalTraceChunk(BaseModel):
    """Top-k retrieved chunk trace item."""

    chunk_id: str
    score: float
    backend: Optional[str] = None  # sql|graph|bm25|azure_search|pgvector
    rank: int = Field(..., ge=1)
    sources: list[str] = Field(default_factory=list)


class RetrievalTrace(BaseModel):
    """Evidence retrieval trace."""

    top_k_chunks: list[RetrievalTraceChunk] = Field(default_factory=list)
    top_k: int = Field(default=10, ge=1)


class GraphTracePath(BaseModel):
    """A graph path trace."""

    nodes: list[str] = Field(default_factory=list)
    edges: list[str] = Field(default_factory=list)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)


class GraphTraceUsedEdgeSpan(EvidenceSpanRef):
    """A justification span for a used semantic edge."""

    chunk_id: str = Field(..., description="Chunk ID that contains the justification span")


class GraphTraceUsedEdge(BaseModel):
    """
    A semantic edge that the answer *actually used* (not merely retrieved).

    Required by stakeholder readiness gates:
    - edge_id, from_node, to_node, relation_type
    - justification_spans: >= 1 span with quote + offsets
    """

    edge_id: str
    from_node: str
    to_node: str
    relation_type: str
    justification_spans: list[GraphTraceUsedEdgeSpan] = Field(default_factory=list)


class ArgumentInferenceType(str, Enum):
    """Deterministic inference classification for argument chains."""

    DIRECT_QUOTE = "direct_quote"
    MULTI_SPAN_ENTAILMENT = "multi_span_entailment"


class ArgumentChain(BaseModel):
    """
    Deterministic, inspectable argument chain (no CoT).

    This is the "argument quality" layer on top of used_edges:
    - claim_ar: what the system asserts (tied to an edge)
    - evidence_spans: 1-N spans supporting the claim
    - inference_type: direct_quote vs multi_span_entailment
    - boundary_ar: quoted if present, else "غير منصوص عليه في الإطار"
    """

    edge_id: str
    relation_type: str
    from_node: str
    to_node: str

    claim_ar: str
    inference_type: ArgumentInferenceType

    evidence_spans: list[EvidenceSpanRef] = Field(default_factory=list)
    boundary_ar: str = "غير منصوص عليه في الإطار"
    boundary_spans: list[EvidenceSpanRef] = Field(default_factory=list)


class GraphTrace(BaseModel):
    """Graph traversal trace."""

    nodes: list[str] = Field(default_factory=list)
    edges: list[str] = Field(default_factory=list)
    paths: list[GraphTracePath] = Field(default_factory=list)
    used_edges: list[GraphTraceUsedEdge] = Field(default_factory=list)
    argument_chains: list[ArgumentChain] = Field(default_factory=list)


class AlmuhasbiTrace(BaseModel):
    """Safe, non-CoT summary of Muḥāsibī/AlMuhasbi value-add."""

    changed_summary_ar: str = ""
    reasons_ar: list[str] = Field(default_factory=list)
    supported_by_chunk_ids: list[str] = Field(default_factory=list)


class ClaimEvidenceBinding(BaseModel):
    """Explicit mapping from a claim to supporting evidence spans."""

    supporting_spans: list[EvidenceSpanRef] = Field(default_factory=list)


class EvalClaim(BaseModel):
    """An atomic claim extracted/declared by the system, with explicit support policy."""

    claim_id: str
    text_ar: str
    text_en: Optional[str] = None

    support_strength: ClaimSupportStrength
    support_policy: ClaimSupportPolicy

    evidence: ClaimEvidenceBinding = Field(default_factory=ClaimEvidenceBinding)

    requires_evidence: bool = True
    # Reason: for LLM_ONLY_UNGROUNDED we treat claims as not requiring evidence for core KPIs.

    claim_type: Optional[
        Literal[
            "definition",
            "fact",
            "relationship",
            "recommendation",
            "abstention_reason",
            "meta",
        ]
    ] = None


class EvalRunMetadata(BaseModel):
    """Run-level metadata (pinned, deterministic)."""

    run_id: str
    dataset_id: str
    dataset_version: str
    dataset_sha256: str

    seed: int

    # Pinned model/prompt identifiers (strings) used for reproducibility.
    llm_provider: Optional[str] = None
    llm_model: Optional[str] = None
    llm_deployment: Optional[str] = None

    prompts_version: str = "v1"

    code_commit: Optional[str] = None
    generated_at_utc: str


class MechanismLoopTrace(BaseModel):
    """Trace of a causal loop used in the answer."""
    
    loop_id: str
    loop_type: str  # reinforcing|balancing
    nodes: list[str] = Field(default_factory=list)
    node_labels_ar: list[str] = Field(default_factory=list)
    polarities: list[int] = Field(default_factory=list)
    edge_ids: list[str] = Field(default_factory=list)
    evidence_spans: list[EvidenceSpanRef] = Field(default_factory=list)


class InterventionStepTrace(BaseModel):
    """Trace of an intervention step."""
    
    target_node: str
    target_label_ar: str
    mechanism_reason_ar: str
    citations: list[EvidenceSpanRef] = Field(default_factory=list)
    expected_impacts: list[str] = Field(default_factory=list)


class InterventionPlanTrace(BaseModel):
    """Trace of an intervention plan."""
    
    goal_ar: str
    steps: list[InterventionStepTrace] = Field(default_factory=list)
    leading_indicators: list[dict[str, Any]] = Field(default_factory=list)
    risk_of_imbalance: list[dict[str, Any]] = Field(default_factory=list)


class SimulationTrace(BaseModel):
    """Trace of a counterfactual simulation."""
    
    scenario_ar: str = ""
    initial_node: str = ""
    propagation_steps: list[dict[str, Any]] = Field(default_factory=list)
    affected_nodes_count: int = 0
    label_ar: str = "محاكاة تقريبية وفق روابط الإطار"


class MechanismTrace(BaseModel):
    """World model mechanism trace."""
    
    loops: list[MechanismLoopTrace] = Field(default_factory=list)
    interventions: list[InterventionPlanTrace] = Field(default_factory=list)
    simulation_summary: list[SimulationTrace] = Field(default_factory=list)
    covered_pillars: list[str] = Field(default_factory=list)
    loops_count: int = 0
    interventions_count: int = 0


class EvalOutputRow(BaseModel):
    """One JSONL row for one question in one mode."""

    id: str
    mode: EvalMode
    question: str

    answer_ar: str
    answer_en: Optional[str] = None

    claims: list[EvalClaim] = Field(default_factory=list)
    citations: list[EvalCitation] = Field(default_factory=list)

    retrieval_trace: RetrievalTrace = Field(default_factory=RetrievalTrace)
    graph_trace: GraphTrace = Field(default_factory=GraphTrace)
    mechanism_trace: Optional[MechanismTrace] = None  # World model trace
    almuhasbi_trace: Optional[AlmuhasbiTrace] = None

    abstained: bool = False
    abstain_reason: Optional[str] = None

    latency_ms: int = Field(default=0, ge=0)

    # Free-form extra for debugging (must be deterministic and safe)
    debug: dict[str, Any] = Field(default_factory=dict)
