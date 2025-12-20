"""World Model package for system dynamics graphs and causal reasoning.

This package provides:
- Mechanism nodes (anchored to existing pillar/core_value/sub_value hierarchy)
- Mechanism edges with signed polarity (+1/-1)
- Causal loop detection and classification (reinforcing/balancing)
- Intervention planning with framework-bound steps
- Counterfactual simulation with evidence-based weights
- Global synthesis answer composition

Key principles:
- No duplicate ontology: mechanism_node uses ref_kind + ref_id to anchor
- Polarity as signed integer: product of signs determines loop type
- No ungrounded summaries: loop summaries generated on-the-fly from edge spans
- Framework-bound interventions: steps must map to existing nodes
"""

from apps.api.core.world_model.schemas import (
    MechanismNode,
    MechanismEdge,
    MechanismEdgeSpan,
    FeedbackLoop,
    DetectedLoop,
    InterventionStep,
    InterventionPlan,
    SimulationResult,
    RefKind,
    RelationType,
    RELATION_POLARITY_DEFAULTS,
    compute_loop_type,
    get_default_polarity,
    compute_edge_confidence,
    validate_no_medical_claims,
)

from apps.api.core.world_model.loop_reasoner import (
    detect_loops,
    retrieve_relevant_loops,
    compute_loop_relevance_score,
)

from apps.api.core.world_model.intervention_planner import (
    compute_intervention_plan,
    validate_intervention_plan,
)

from apps.api.core.world_model.simulator import (
    simulate_change,
    simulate_what_if,
    propagate_change,
)

from apps.api.core.world_model.composer import (
    build_world_model_plan,
    compose_global_synthesis_answer,
    WorldModelPlan,
)

__all__ = [
    # Schemas
    "MechanismNode",
    "MechanismEdge",
    "MechanismEdgeSpan",
    "FeedbackLoop",
    "DetectedLoop",
    "InterventionStep",
    "InterventionPlan",
    "SimulationResult",
    "RefKind",
    "RelationType",
    "RELATION_POLARITY_DEFAULTS",
    "compute_loop_type",
    "get_default_polarity",
    "compute_edge_confidence",
    "validate_no_medical_claims",
    # Loop reasoner
    "detect_loops",
    "retrieve_relevant_loops",
    "compute_loop_relevance_score",
    # Intervention planner
    "compute_intervention_plan",
    "validate_intervention_plan",
    # Simulator
    "simulate_change",
    "simulate_what_if",
    "propagate_change",
    # Composer
    "build_world_model_plan",
    "compose_global_synthesis_answer",
    "WorldModelPlan",
]
