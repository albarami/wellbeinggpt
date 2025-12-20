"""
UI-facing schemas (additive; does not change FinalResponse contract).

Reason:
- `FinalResponse` is an authoritative contract (see apps/api/core/schemas.py) and must not change.
- The production UI needs additional, traceable fields: spans, graph trace, contract outcome, request_id.
"""

from __future__ import annotations

from pydantic import BaseModel, Field
from typing import Any, Optional

from apps.api.core.schemas import FinalResponse


class SpanResolutionStatus(str):
    """Resolution status for a requested text span."""

    RESOLVED = "resolved"
    UNRESOLVED = "unresolved"


class CitationSpan(BaseModel):
    """
    UI citation span resolved against stored chunk text.

    Hard gate:
    - If offsets cannot be resolved reliably, set span_start/span_end to null and
      span_resolution_status to "unresolved".
    """

    chunk_id: str = Field(..., min_length=1)
    source_id: str = Field(..., min_length=1, description="source_doc_id")

    quote: str = Field(..., description="Excerpt rendered and highlightable in the UI")
    span_start: Optional[int] = Field(default=None, ge=0)
    span_end: Optional[int] = Field(default=None, ge=0)

    source_anchor: Optional[str] = None
    entity_type: Optional[str] = None
    entity_id: Optional[str] = None

    span_resolution_status: str = Field(default=SpanResolutionStatus.UNRESOLVED)
    span_resolution_method: str = Field(default="unresolved")


class UsedEdge(BaseModel):
    """UI-safe representation of a relied-upon semantic edge."""

    edge_id: str
    from_node: str
    to_node: str
    relation_type: str
    justification_spans: list[dict[str, Any]] = Field(default_factory=list)


class ArgumentChain(BaseModel):
    """UI-safe deterministic argument chain (no chain-of-thought)."""

    edge_id: str
    relation_type: str
    from_node: str
    to_node: str
    claim_ar: str
    inference_type: str
    evidence_spans: list[dict[str, Any]] = Field(default_factory=list)
    boundary_ar: str
    boundary_spans: list[dict[str, Any]] = Field(default_factory=list)


class GraphTrace(BaseModel):
    used_edges: list[UsedEdge] = Field(default_factory=list)
    argument_chains: list[ArgumentChain] = Field(default_factory=list)


# =============================================================================
# World Model Mechanism Trace (loops, interventions, simulations)
# =============================================================================


class MechanismLoopUI(BaseModel):
    """UI-safe representation of a causal loop.
    
    Note: summary_ar is generated on-the-fly from edge spans (adjustment #5).
    """
    
    loop_id: str
    loop_type: str = Field(..., description="reinforcing|balancing")
    nodes: list[str] = Field(default_factory=list, description="ref_kind:ref_id format")
    node_labels_ar: list[str] = Field(default_factory=list, description="Display labels")
    polarities: list[int] = Field(default_factory=list, description="+1 or -1 per edge")
    evidence_spans: list[dict[str, Any]] = Field(default_factory=list)
    summary_ar: str = Field(default="", description="Generated on-the-fly from spans")


class InterventionStepUI(BaseModel):
    """UI-safe representation of an intervention step."""
    
    target_node: str = Field(..., description="ref_kind:ref_id")
    target_label_ar: str
    mechanism_reason_ar: str
    citations: list[dict[str, Any]] = Field(default_factory=list)
    expected_impacts: list[str] = Field(default_factory=list)


class InterventionPlanUI(BaseModel):
    """UI-safe representation of an intervention plan."""
    
    goal_ar: str
    steps: list[InterventionStepUI] = Field(default_factory=list)
    leading_indicators: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Each includes 'source': 'framework'|'غير منصوص'"
    )
    risk_of_imbalance: list[dict[str, Any]] = Field(default_factory=list)


class SimulationSummaryUI(BaseModel):
    """UI-safe representation of a simulation result."""
    
    scenario_ar: str = ""
    initial_node: str = ""
    propagation_steps: list[dict[str, Any]] = Field(default_factory=list)
    affected_nodes_count: int = 0
    label_ar: str = "محاكاة تقريبية وفق روابط الإطار"


class MechanismTrace(BaseModel):
    """World Model mechanism trace for UI.
    
    Contains:
    - Detected causal loops (reinforcing/balancing)
    - Intervention plans with framework-bound steps
    - Simulation summaries (if counterfactual analysis was performed)
    """
    
    loops: list[MechanismLoopUI] = Field(default_factory=list)
    interventions: list[InterventionPlanUI] = Field(default_factory=list)
    simulation_summary: list[SimulationSummaryUI] = Field(default_factory=list)
    
    # Summary counts for quick UI display
    loops_count: int = 0
    interventions_count: int = 0
    covered_pillars: list[str] = Field(default_factory=list)


class MuhasibiTraceEvent(BaseModel):
    """
    Safe Muḥāsibī flow event (no chain-of-thought).

    This mirrors the safe trace fields from /ask/trace, but is exposed inside /ask/ui
    so the UI can show the pipeline flow even in natural_chat mode.
    """

    state: str
    elapsed_s: float
    mode: str
    language: str
    detected_entities_count: Optional[int] = None
    keywords_count: Optional[int] = None
    listen_summary_ar: Optional[str] = None
    ultimate_goal_ar: Optional[str] = None
    constraints_count: Optional[int] = None
    path_steps: Optional[list[str]] = None
    evidence_packets_count: Optional[int] = None
    has_definition: Optional[bool] = None
    has_evidence: Optional[bool] = None
    not_found: Optional[bool] = None
    issues: Optional[list[str]] = None
    confidence: Optional[str] = None
    citations_count: Optional[int] = None
    reflection_added: Optional[bool] = None
    done: Optional[bool] = None


class AskUiResponse(BaseModel):
    """
    Additive response for production UI.

    - `final` contains the unchanged authoritative FinalResponse.
    - Other fields provide grounded traces and observability.
    """

    request_id: str
    latency_ms: int = Field(..., ge=0)
    mode_used: str
    engine_used: str

    contract_outcome: str = Field(..., description="PASS_FULL|PASS_PARTIAL|FAIL")
    contract_reasons: list[str] = Field(default_factory=list)
    contract_applicable: bool = True

    abstain_reason: Optional[str] = None

    citations_spans: list[CitationSpan] = Field(default_factory=list)
    graph_trace: GraphTrace = Field(default_factory=GraphTrace)
    mechanism_trace: MechanismTrace = Field(default_factory=MechanismTrace)
    muhasibi_trace: list[MuhasibiTraceEvent] = Field(default_factory=list)

    truncated_fields: dict[str, Any] = Field(default_factory=dict)
    original_counts: dict[str, Any] = Field(default_factory=dict)

    final: FinalResponse

