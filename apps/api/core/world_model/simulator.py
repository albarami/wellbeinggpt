"""Counterfactual simulator for World Model.

This module provides deterministic propagation simulation:
- Weights derived from evidence density (span count + diversity)
- BFS-based propagation with damping
- Explanatory output labeled as approximate

Key principle: This is for explanation, not numeric truth.
All outputs are labeled as "محاكاة تقريبية وفق روابط الإطار".
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from apps.api.core.world_model.schemas import (
    SimulationResult,
    compute_edge_confidence,
)
from apps.api.core.world_model.loop_reasoner import (
    GraphEdge,
    GraphNode,
    load_mechanism_graph,
)


# Damping factor to prevent runaway propagation
DAMPING_FACTOR = 0.7

# Minimum delta to continue propagation
MIN_DELTA_THRESHOLD = 0.01

# Default initial value for nodes
DEFAULT_NODE_VALUE = 0.5


@dataclass
class PropagationStep:
    """A single step in the propagation simulation."""
    step_number: int
    node_id: str
    node_label_ar: str
    delta: float
    new_value: float
    via_edge_id: str
    via_relation: str


def compute_edge_weight(edge: GraphEdge) -> float:
    """Compute evidence-based weight for an edge.
    
    Per recommendation A: weight derived from:
    - Number of justification spans (more = higher)
    - Span diversity (different chunks = higher)
    - Polarity (+1 or -1)
    - Confidence
    
    Args:
        edge: The graph edge
        
    Returns:
        Signed weight in [-0.5, 0.5] range approximately
    """
    span_count = len(edge.spans)
    chunk_ids = set(sp.get("chunk_id", "") for sp in edge.spans)
    chunk_diversity = len(chunk_ids)
    
    # Base weight from evidence
    base = 0.1 + min(0.4, span_count * 0.1 + chunk_diversity * 0.05)
    
    # Apply polarity and confidence
    weight = base * edge.polarity * edge.confidence
    
    return weight


def propagate_change(
    nodes_by_id: dict[str, GraphNode],
    edges: list[GraphEdge],
    changed_node_id: str,
    change_magnitude: float,
    max_steps: int = 5,
) -> SimulationResult:
    """Deterministic propagation using BFS with damping.
    
    At each step:
    1. For each outgoing edge from affected nodes
    2. delta = current_value * edge_weight * damping_factor
    3. Accumulate deltas (don't double-count)
    4. Stop when deltas < threshold or max_steps reached
    
    Args:
        nodes_by_id: Map of node IDs to GraphNode
        edges: List of all edges
        changed_node_id: ID of the node being changed
        change_magnitude: Amount of change (+/- value)
        max_steps: Maximum propagation steps
        
    Returns:
        SimulationResult with propagation trace
    """
    # Initialize state
    initial_state: dict[str, float] = {}
    current_state: dict[str, float] = {}
    
    for node_id in nodes_by_id:
        initial_state[node_id] = DEFAULT_NODE_VALUE
        current_state[node_id] = DEFAULT_NODE_VALUE
    
    # Apply initial change
    if changed_node_id in current_state:
        current_state[changed_node_id] = min(1.0, max(0.0, 
            current_state[changed_node_id] + change_magnitude
        ))
    
    # Build adjacency for outgoing edges
    outgoing: dict[str, list[GraphEdge]] = defaultdict(list)
    for edge in edges:
        outgoing[edge.from_node].append(edge)
    
    # Track propagation steps
    propagation_steps: list[PropagationStep] = []
    
    # BFS propagation
    affected_nodes = {changed_node_id}
    step_number = 0
    
    for _ in range(max_steps):
        step_number += 1
        next_affected: set[str] = set()
        deltas_this_step: dict[str, float] = {}
        
        for source_id in affected_nodes:
            source_value = current_state.get(source_id, DEFAULT_NODE_VALUE)
            
            for edge in outgoing.get(source_id, []):
                target_id = edge.to_node
                if target_id not in nodes_by_id:
                    continue
                
                # Compute delta
                weight = compute_edge_weight(edge)
                delta = source_value * weight * DAMPING_FACTOR
                
                # Skip negligible changes
                if abs(delta) < MIN_DELTA_THRESHOLD:
                    continue
                
                # Accumulate delta (don't double-count same target)
                if target_id in deltas_this_step:
                    deltas_this_step[target_id] += delta
                else:
                    deltas_this_step[target_id] = delta
                
                # Record propagation step
                target_node = nodes_by_id[target_id]
                propagation_steps.append(PropagationStep(
                    step_number=step_number,
                    node_id=target_id,
                    node_label_ar=target_node.label_ar,
                    delta=round(delta, 4),
                    new_value=round(current_state[target_id] + delta, 4),
                    via_edge_id=edge.id,
                    via_relation=edge.relation_type,
                ))
                
                next_affected.add(target_id)
        
        # Apply deltas
        for target_id, delta in deltas_this_step.items():
            current_state[target_id] = min(1.0, max(0.0, 
                current_state[target_id] + delta
            ))
        
        # Check if we should continue
        if not next_affected:
            break
        
        affected_nodes = next_affected
    
    # Build final state (only nodes that changed)
    final_state: dict[str, float] = {}
    for node_id, value in current_state.items():
        if abs(value - initial_state[node_id]) > MIN_DELTA_THRESHOLD:
            final_state[node_id] = round(value, 4)
    
    return SimulationResult(
        initial_state={k: round(v, 4) for k, v in initial_state.items()},
        final_state=final_state,
        propagation_steps=[
            {
                "step": ps.step_number,
                "node": ps.node_id,
                "node_label_ar": ps.node_label_ar,
                "delta": ps.delta,
                "new_value": ps.new_value,
                "via_edge": ps.via_edge_id,
                "via_relation": ps.via_relation,
            }
            for ps in propagation_steps
        ],
        label_ar="محاكاة تقريبية وفق روابط الإطار",
    )


async def simulate_change(
    session: AsyncSession,
    changed_node_ref: str,  # format: "ref_kind:ref_id"
    change_magnitude: float,
    max_steps: int = 5,
) -> SimulationResult:
    """Simulate propagation of a change through the mechanism graph.
    
    Args:
        session: Database session
        changed_node_ref: Reference to changed node (e.g., "pillar:P001")
        change_magnitude: Amount of change (+/- value, typically ±0.2)
        max_steps: Maximum propagation steps
        
    Returns:
        SimulationResult with propagation trace
    """
    # Load graph
    nodes_by_id, edges = await load_mechanism_graph(session)
    
    if not nodes_by_id or not edges:
        return SimulationResult(
            initial_state={},
            final_state={},
            propagation_steps=[],
            label_ar="محاكاة تقريبية وفق روابط الإطار - لا توجد بيانات كافية",
        )
    
    # Find the changed node
    changed_node_id = None
    for node in nodes_by_id.values():
        if f"{node.ref_kind}:{node.ref_id}" == changed_node_ref:
            changed_node_id = node.id
            break
    
    if not changed_node_id:
        return SimulationResult(
            initial_state={},
            final_state={},
            propagation_steps=[],
            label_ar=f"محاكاة تقريبية - لم يُعثر على العقدة: {changed_node_ref}",
        )
    
    # Run simulation
    return propagate_change(
        nodes_by_id=nodes_by_id,
        edges=edges,
        changed_node_id=changed_node_id,
        change_magnitude=change_magnitude,
        max_steps=max_steps,
    )


async def simulate_what_if(
    session: AsyncSession,
    scenario_ar: str,
    pillar_changes: dict[str, float],  # pillar_id -> change magnitude
    max_steps: int = 5,
) -> dict[str, Any]:
    """Simulate a "what if" scenario with multiple pillar changes.
    
    Args:
        session: Database session
        scenario_ar: Arabic description of scenario
        pillar_changes: Map of pillar IDs to change magnitudes
        max_steps: Maximum propagation steps
        
    Returns:
        Dictionary with scenario results
    """
    # Load graph once
    nodes_by_id, edges = await load_mechanism_graph(session)
    
    if not nodes_by_id or not edges:
        return {
            "scenario_ar": scenario_ar,
            "results": [],
            "combined_impacts": {},
            "label_ar": "محاكاة تقريبية وفق روابط الإطار - لا توجد بيانات كافية",
        }
    
    results: list[dict[str, Any]] = []
    combined_impacts: dict[str, float] = {}
    
    for pillar_id, magnitude in pillar_changes.items():
        # Find pillar node
        pillar_node_id = None
        for node in nodes_by_id.values():
            if node.ref_kind == "pillar" and node.ref_id == pillar_id:
                pillar_node_id = node.id
                break
        
        if not pillar_node_id:
            continue
        
        # Simulate this change
        sim_result = propagate_change(
            nodes_by_id=nodes_by_id,
            edges=edges,
            changed_node_id=pillar_node_id,
            change_magnitude=magnitude,
            max_steps=max_steps,
        )
        
        results.append({
            "pillar_id": pillar_id,
            "change_magnitude": magnitude,
            "propagation_steps": sim_result.propagation_steps,
            "final_state": sim_result.final_state,
        })
        
        # Accumulate impacts
        for node_id, value in sim_result.final_state.items():
            if node_id in combined_impacts:
                combined_impacts[node_id] = max(combined_impacts[node_id], value)
            else:
                combined_impacts[node_id] = value
    
    return {
        "scenario_ar": scenario_ar,
        "results": results,
        "combined_impacts": combined_impacts,
        "label_ar": "محاكاة تقريبية وفق روابط الإطار",
    }


def simulation_result_to_summary_ar(
    result: SimulationResult,
    nodes_by_id: dict[str, GraphNode],
) -> str:
    """Generate Arabic summary of simulation result.
    
    Args:
        result: The simulation result
        nodes_by_id: Map of node IDs to GraphNode
        
    Returns:
        Arabic summary text
    """
    if not result.propagation_steps:
        return "لم تُظهر المحاكاة انتشارًا ملحوظًا."
    
    # Count affected nodes by type
    affected_pillars: list[str] = []
    affected_values: list[str] = []
    
    for step in result.propagation_steps:
        node_id = step.get("node") or step.get("node_id", "")
        if node_id in nodes_by_id:
            node = nodes_by_id[node_id]
            label = node.label_ar
            
            if node.ref_kind == "pillar" and label not in affected_pillars:
                affected_pillars.append(label)
            elif label not in affected_values:
                affected_values.append(label)
    
    parts: list[str] = []
    
    if affected_pillars:
        parts.append(f"الركائز المتأثرة: {', '.join(affected_pillars[:3])}")
    
    if affected_values:
        parts.append(f"القيم المتأثرة: {', '.join(affected_values[:3])}")
    
    parts.append(f"عدد خطوات الانتشار: {len(result.propagation_steps)}")
    
    summary = " | ".join(parts)
    return f"{result.label_ar} — {summary}"
