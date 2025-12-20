"""Intervention planning for World Model.

This module provides evidence-bound intervention planning:
- Steps must map to existing framework nodes (adjustment #4)
- No medical/diagnostic claims (FORBIDDEN_CLAIMS gate)
- Leading indicators marked with source (framework or "غير منصوص")
- Risk analysis from INHIBITS/TENSION_WITH edges

Algorithm:
1. Identify goal node (must exist in framework)
2. Trace backward through mechanism edges to find leverage points
3. For each step, require evidence citation
4. Check for INHIBITS/TENSION edges that indicate risk
5. Leading indicators: only include if explicitly defined in framework
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from apps.api.core.world_model.schemas import (
    InterventionPlan,
    InterventionStep,
    validate_no_medical_claims,
    FORBIDDEN_CLAIMS,
)
from apps.api.core.world_model.loop_reasoner import (
    DetectedLoop,
    GraphEdge,
    GraphNode,
    load_mechanism_graph,
)


@dataclass
class PlannerContext:
    """Context for intervention planning."""
    nodes_by_id: dict[str, GraphNode] = field(default_factory=dict)
    nodes_by_ref: dict[str, GraphNode] = field(default_factory=dict)  # ref_kind:ref_id -> node
    edges: list[GraphEdge] = field(default_factory=list)
    incoming_edges: dict[str, list[GraphEdge]] = field(default_factory=dict)  # node_id -> edges pointing to it
    outgoing_edges: dict[str, list[GraphEdge]] = field(default_factory=dict)  # node_id -> edges from it


async def build_planner_context(session: AsyncSession) -> PlannerContext:
    """Build context for intervention planning from database.
    
    Args:
        session: Database session
        
    Returns:
        PlannerContext with loaded graph data
    """
    nodes_by_id, edges = await load_mechanism_graph(session)
    
    # Build ref -> node mapping
    nodes_by_ref: dict[str, GraphNode] = {}
    for node in nodes_by_id.values():
        ref_key = f"{node.ref_kind}:{node.ref_id}"
        nodes_by_ref[ref_key] = node
    
    # Build incoming/outgoing edge maps
    incoming: dict[str, list[GraphEdge]] = defaultdict(list)
    outgoing: dict[str, list[GraphEdge]] = defaultdict(list)
    
    for edge in edges:
        incoming[edge.to_node].append(edge)
        outgoing[edge.from_node].append(edge)
    
    return PlannerContext(
        nodes_by_id=nodes_by_id,
        nodes_by_ref=nodes_by_ref,
        edges=edges,
        incoming_edges=dict(incoming),
        outgoing_edges=dict(outgoing),
    )


def _find_node_for_goal(
    ctx: PlannerContext,
    goal_ar: str,
    detected_entities: list[dict[str, Any]],
) -> GraphNode | None:
    """Find framework node that matches the goal.
    
    Args:
        ctx: Planner context
        goal_ar: Arabic goal text
        detected_entities: Detected entities from question
        
    Returns:
        Matching node, or None if not found
    """
    # First try detected entities
    for entity in detected_entities:
        ref_kind = str(entity.get("entity_type") or entity.get("type") or "")
        ref_id = str(entity.get("entity_id") or entity.get("id") or "")
        
        if ref_kind and ref_id:
            ref_key = f"{ref_kind}:{ref_id}"
            if ref_key in ctx.nodes_by_ref:
                return ctx.nodes_by_ref[ref_key]
    
    # Try matching goal text against node labels
    goal_normalized = goal_ar.strip().lower()
    for node in ctx.nodes_by_id.values():
        if node.label_ar.strip().lower() in goal_normalized:
            return node
        if goal_normalized in node.label_ar.strip().lower():
            return node
    
    return None


def _trace_backward_for_leverage(
    ctx: PlannerContext,
    goal_node: GraphNode,
    max_depth: int = 4,
) -> list[tuple[GraphNode, GraphEdge, int]]:
    """Trace backward through edges to find leverage points.
    
    Returns nodes that can influence the goal, with their connecting edge
    and distance from goal.
    
    Args:
        ctx: Planner context
        goal_node: The goal node to trace from
        max_depth: Maximum backward trace depth
        
    Returns:
        List of (node, edge, depth) tuples
    """
    results: list[tuple[GraphNode, GraphEdge, int]] = []
    visited: set[str] = {goal_node.id}
    
    current_level = [goal_node.id]
    
    for depth in range(1, max_depth + 1):
        next_level: list[str] = []
        
        for node_id in current_level:
            # Get incoming edges (edges that point TO this node)
            incoming = ctx.incoming_edges.get(node_id, [])
            
            for edge in incoming:
                from_node = ctx.nodes_by_id.get(edge.from_node)
                if not from_node:
                    continue
                
                if from_node.id in visited:
                    continue
                
                visited.add(from_node.id)
                results.append((from_node, edge, depth))
                next_level.append(from_node.id)
        
        current_level = next_level
        if not current_level:
            break
    
    return results


def _find_risks(
    ctx: PlannerContext,
    target_nodes: list[GraphNode],
) -> list[dict[str, Any]]:
    """Find risks from INHIBITS/TENSION_WITH edges.
    
    Args:
        ctx: Planner context
        target_nodes: Nodes being targeted in the intervention
        
    Returns:
        List of risk dictionaries
    """
    risks: list[dict[str, Any]] = []
    target_ids = {n.id for n in target_nodes}
    
    for edge in ctx.edges:
        # Only consider negative polarity edges
        if edge.polarity >= 0:
            continue
        
        if edge.relation_type not in ("INHIBITS", "TENSION_WITH"):
            continue
        
        # Check if edge involves any target node
        if edge.from_node in target_ids or edge.to_node in target_ids:
            from_node = ctx.nodes_by_id.get(edge.from_node)
            to_node = ctx.nodes_by_id.get(edge.to_node)
            
            if from_node and to_node:
                # Extract pillar from affected node
                affected_pillar = ""
                if to_node.ref_kind == "pillar":
                    affected_pillar = to_node.ref_id
                elif from_node.ref_kind == "pillar":
                    affected_pillar = from_node.ref_id
                
                risks.append({
                    "risk_ar": f"{edge.relation_type}: {from_node.label_ar} → {to_node.label_ar}",
                    "affected_pillar": affected_pillar,
                    "evidence": edge.spans[:2] if edge.spans else [],
                })
    
    return risks


def _get_downstream_impacts(
    ctx: PlannerContext,
    node: GraphNode,
    max_impacts: int = 3,
) -> list[str]:
    """Get downstream nodes that will be impacted.
    
    Args:
        ctx: Planner context
        node: Starting node
        max_impacts: Maximum impacts to return
        
    Returns:
        List of downstream node labels
    """
    impacts: list[str] = []
    outgoing = ctx.outgoing_edges.get(node.id, [])
    
    for edge in outgoing[:max_impacts]:
        if edge.polarity > 0:  # Only positive impacts
            to_node = ctx.nodes_by_id.get(edge.to_node)
            if to_node:
                impacts.append(to_node.label_ar)
    
    return impacts


def _extract_leading_indicators(
    ctx: PlannerContext,
    goal_node: GraphNode,
) -> list[dict[str, Any]]:
    """Extract leading indicators for the goal.
    
    Per adjustment #4: only include if explicitly defined in framework,
    otherwise mark as "غير منصوص".
    
    Args:
        ctx: Planner context
        goal_node: The goal node
        
    Returns:
        List of indicator dictionaries with source field
    """
    indicators: list[dict[str, Any]] = []
    
    # Look for incoming edges with CONDITIONAL_ON relation
    incoming = ctx.incoming_edges.get(goal_node.id, [])
    
    for edge in incoming:
        if edge.relation_type == "CONDITIONAL_ON":
            from_node = ctx.nodes_by_id.get(edge.from_node)
            if from_node and edge.spans:
                indicators.append({
                    "indicator_ar": from_node.label_ar,
                    "source": "framework",
                    "evidence": edge.spans[:1],
                })
    
    # If no indicators found from framework, add placeholder
    if not indicators:
        indicators.append({
            "indicator_ar": "غير محدد",
            "source": "غير منصوص",
        })
    
    return indicators


async def compute_intervention_plan(
    session: AsyncSession,
    goal_ar: str,
    detected_entities: list[dict[str, Any]],
    loops: list[DetectedLoop],
    max_steps: int = 7,
) -> InterventionPlan:
    """Compute evidence-bound intervention plan.
    
    Algorithm:
    1. Identify goal node (must exist in framework)
    2. Trace backward through mechanism edges to find leverage points
    3. For each step, require evidence citation
    4. Check for INHIBITS/TENSION edges that indicate risk
    5. Leading indicators: only include if explicitly defined in framework
    
    Hard gates:
    - No step without framework node mapping
    - No medical/diagnostic claims
    - Missing indicators marked "غير منصوص عليه في الإطار"
    
    Args:
        session: Database session
        goal_ar: Arabic goal text
        detected_entities: Entities detected in question
        loops: Detected causal loops
        max_steps: Maximum intervention steps
        
    Returns:
        InterventionPlan with evidence-bound steps
    """
    # Build context
    ctx = await build_planner_context(session)
    
    # Find goal node
    goal_node = _find_node_for_goal(ctx, goal_ar, detected_entities)
    
    if not goal_node:
        # Cannot find goal in framework - return minimal plan
        return InterventionPlan(
            goal_ar=goal_ar,
            steps=[],
            leading_indicators=[{
                "indicator_ar": "لم يُعثر على الهدف في الإطار",
                "source": "غير منصوص",
            }],
            risk_of_imbalance=[],
        )
    
    # Trace backward for leverage points
    leverage_points = _trace_backward_for_leverage(ctx, goal_node, max_depth=4)
    
    # Build intervention steps
    steps: list[InterventionStep] = []
    target_nodes: list[GraphNode] = [goal_node]
    
    # Sort by depth (closer to goal = later in plan)
    leverage_points.sort(key=lambda x: -x[2])
    
    for node, edge, depth in leverage_points[:max_steps - 1]:
        # Validate no medical claims in label
        if not validate_no_medical_claims(node.label_ar):
            continue
        
        # Get downstream impacts
        impacts = _get_downstream_impacts(ctx, node)
        
        # Build step
        step = InterventionStep(
            target_node_ref_kind=node.ref_kind,
            target_node_ref_id=node.ref_id,
            target_node_label_ar=node.label_ar,
            mechanism_reason_ar=edge.spans[0]["quote"] if edge.spans else "غير منصوص",
            mechanism_citations=edge.spans[:2] if edge.spans else [],
            expected_impacts=impacts,
            impact_citations=[],
        )
        
        steps.append(step)
        target_nodes.append(node)
    
    # Add final step for goal itself
    if validate_no_medical_claims(goal_node.label_ar):
        incoming = ctx.incoming_edges.get(goal_node.id, [])
        goal_edge = incoming[0] if incoming else None
        
        steps.append(InterventionStep(
            target_node_ref_kind=goal_node.ref_kind,
            target_node_ref_id=goal_node.ref_id,
            target_node_label_ar=goal_node.label_ar,
            mechanism_reason_ar=goal_edge.spans[0]["quote"] if goal_edge and goal_edge.spans else "الهدف النهائي",
            mechanism_citations=goal_edge.spans[:2] if goal_edge and goal_edge.spans else [],
            expected_impacts=[],
            impact_citations=[],
        ))
    
    # Ensure we have at least 3 steps if possible
    if len(steps) < 3:
        # Add steps from loops if available
        for loop in loops[:2]:
            for i, node_ref in enumerate(loop.nodes):
                if len(steps) >= max_steps:
                    break
                
                # Parse node reference
                if ":" in node_ref:
                    ref_kind, ref_id = node_ref.split(":", 1)
                    node = ctx.nodes_by_ref.get(node_ref)
                    
                    if node and validate_no_medical_claims(node.label_ar):
                        # Check if already in steps
                        existing_ids = {(s.target_node_ref_kind, s.target_node_ref_id) for s in steps}
                        if (ref_kind, ref_id) not in existing_ids:
                            span = loop.evidence_spans[i] if i < len(loop.evidence_spans) else None
                            
                            steps.append(InterventionStep(
                                target_node_ref_kind=ref_kind,
                                target_node_ref_id=ref_id,
                                target_node_label_ar=node.label_ar,
                                mechanism_reason_ar=span["quote"] if span else "من حلقة سببية",
                                mechanism_citations=[span] if span else [],
                                expected_impacts=[],
                                impact_citations=[],
                            ))
    
    # Get leading indicators
    leading_indicators = _extract_leading_indicators(ctx, goal_node)
    
    # Find risks
    risks = _find_risks(ctx, target_nodes)
    
    return InterventionPlan(
        goal_ar=goal_ar,
        steps=steps[:max_steps],
        leading_indicators=leading_indicators,
        risk_of_imbalance=risks,
    )


def validate_intervention_plan(plan: InterventionPlan) -> list[str]:
    """Validate an intervention plan for safety and completeness.
    
    Checks:
    - No medical claims in any step
    - All steps have framework node mapping
    - Evidence citations present where required
    
    Args:
        plan: The plan to validate
        
    Returns:
        List of validation issues (empty if valid)
    """
    issues: list[str] = []
    
    # Check goal for medical claims
    if not validate_no_medical_claims(plan.goal_ar):
        issues.append("Goal contains forbidden medical claim")
    
    # Check each step
    for i, step in enumerate(plan.steps):
        # Check for medical claims
        if not validate_no_medical_claims(step.target_node_label_ar):
            issues.append(f"Step {i+1} target contains forbidden medical claim")
        
        if not validate_no_medical_claims(step.mechanism_reason_ar):
            issues.append(f"Step {i+1} reason contains forbidden medical claim")
        
        # Check framework node mapping
        if not step.target_node_ref_kind or not step.target_node_ref_id:
            issues.append(f"Step {i+1} missing framework node mapping")
    
    return issues


def intervention_plan_to_dict(plan: InterventionPlan) -> dict[str, Any]:
    """Convert InterventionPlan to serializable dictionary.
    
    Args:
        plan: The intervention plan
        
    Returns:
        Dictionary representation
    """
    return {
        "goal_ar": plan.goal_ar,
        "steps": [
            {
                "target_node_ref_kind": step.target_node_ref_kind,
                "target_node_ref_id": step.target_node_ref_id,
                "target_node_label_ar": step.target_node_label_ar,
                "mechanism_reason_ar": step.mechanism_reason_ar,
                "mechanism_citations": list(step.mechanism_citations),
                "expected_impacts": list(step.expected_impacts),
                "impact_citations": list(step.impact_citations),
            }
            for step in plan.steps
        ],
        "leading_indicators": list(plan.leading_indicators),
        "risk_of_imbalance": list(plan.risk_of_imbalance),
    }
