"""World Model scoring metrics.

This module provides scoring for world model synthesis answers:
- Loop relevance score (overlap with question entities/pillars)
- Intervention completeness score (steps map to values + citations)
- Mechanism coverage score (edges used / available)
- Boundary completeness score (explicit boundary statements)
- Quote budget compliance score (quote length within limits)

Per recommendation B: Measure mechanism usefulness, not just counts.
"""

from __future__ import annotations

from typing import Any


def compute_loop_relevance_score(
    loops_used: list[dict[str, Any]],
    question_entities: list[dict[str, Any]],
    question_pillars: list[str],
) -> float:
    """Compute relevance score for loops used in answer.
    
    Score = average over loops of (matched_nodes / total_nodes)
    
    Args:
        loops_used: List of loop dicts with 'nodes' field
        question_entities: Entities detected in question
        question_pillars: Pillar IDs detected in question
        
    Returns:
        Score in [0, 1] range
    """
    if not loops_used:
        return 0.0
    
    # Build set of relevant identifiers
    relevant_ids: set[str] = set()
    
    for pillar in question_pillars:
        relevant_ids.add(f"pillar:{pillar}")
        relevant_ids.add(pillar)
    
    for entity in question_entities:
        etype = str(entity.get("entity_type") or entity.get("type") or "")
        eid = str(entity.get("entity_id") or entity.get("id") or "")
        if etype and eid:
            relevant_ids.add(f"{etype}:{eid}")
            relevant_ids.add(eid)
    
    # Score each loop
    loop_scores: list[float] = []
    
    for loop in loops_used:
        nodes = loop.get("nodes", [])
        if not nodes:
            continue
        
        matched = 0
        for node in nodes:
            if node in relevant_ids:
                matched += 1
            # Also check just the ID part (after colon)
            if isinstance(node, str) and ":" in node:
                _, node_id = node.split(":", 1)
                if node_id in relevant_ids:
                    matched += 1
        
        ratio = matched / len(nodes) if nodes else 0.0
        loop_scores.append(ratio)
    
    if not loop_scores:
        return 0.0
    
    return round(sum(loop_scores) / len(loop_scores), 3)


def compute_intervention_completeness_score(
    plan: dict[str, Any],
) -> float:
    """Compute completeness score for intervention plan.
    
    Score components:
    - steps_with_framework_node: steps where target_node exists / total steps
    - steps_with_mechanism_citation: steps with evidence / total steps  
    - impacts_with_citation: cited impacts / total impacts
    
    Final = weighted average (0.4, 0.4, 0.2)
    
    Args:
        plan: Intervention plan dict
        
    Returns:
        Score in [0, 1] range
    """
    steps = plan.get("steps", [])
    
    if not steps:
        return 0.0
    
    # Component 1: Steps with framework node mapping
    steps_with_node = 0
    for step in steps:
        if step.get("target_node_ref_kind") and step.get("target_node_ref_id"):
            steps_with_node += 1
    node_ratio = steps_with_node / len(steps)
    
    # Component 2: Steps with mechanism citation
    steps_with_citation = 0
    for step in steps:
        if step.get("mechanism_citations"):
            steps_with_citation += 1
    citation_ratio = steps_with_citation / len(steps)
    
    # Component 3: Impacts with citation (if applicable)
    total_impacts = 0
    cited_impacts = 0
    for step in steps:
        impacts = step.get("expected_impacts", [])
        total_impacts += len(impacts)
        impact_citations = step.get("impact_citations", [])
        cited_impacts += min(len(impacts), len(impact_citations))
    
    impact_ratio = cited_impacts / total_impacts if total_impacts > 0 else 1.0
    
    # Weighted average
    score = 0.4 * node_ratio + 0.4 * citation_ratio + 0.2 * impact_ratio
    
    return round(score, 3)


def compute_mechanism_coverage_score(
    edges_used: int,
    edges_available: int,
    loops_used: int,
    loops_detected: int,
) -> float:
    """Compute mechanism coverage score.
    
    Measures how much of the available mechanism graph was utilized.
    
    Args:
        edges_used: Number of mechanism edges used in answer
        edges_available: Total mechanism edges available
        loops_used: Number of loops used in answer
        loops_detected: Total loops detected
        
    Returns:
        Score in [0, 1] range
    """
    # Edge utilization (capped at 0.5 - we don't want to use ALL edges)
    edge_ratio = min(0.5, edges_used / edges_available) if edges_available > 0 else 0.0
    
    # Loop utilization (target: use at least 2 loops if available)
    if loops_detected == 0:
        loop_ratio = 1.0  # No loops available, so full score
    else:
        # Score based on using reasonable number of loops (2-5)
        target_loops = min(5, loops_detected)
        loop_ratio = min(1.0, loops_used / target_loops)
    
    # Combined score (loops weighted higher)
    score = 0.3 * (edge_ratio * 2) + 0.7 * loop_ratio
    
    return round(min(1.0, score), 3)


def compute_boundary_completeness_score(
    answer_ar: str,
    boundaries_expected: int = 1,
) -> float:
    """Compute score for explicit boundary statements.
    
    Checks for presence of boundary/limits language in the answer.
    
    Args:
        answer_ar: Arabic answer text
        boundaries_expected: Expected number of boundary statements
        
    Returns:
        Score in [0, 1] range
    """
    if not answer_ar:
        return 0.0
    
    # Boundary markers
    BOUNDARY_MARKERS = [
        "ضوابط", "حدود", "ميزان", "انحراف", "افراط", "إفراط",
        "تفريط", "لا ينبغي", "لا يجوز", "لا يصح", "تحذير",
        "تنبيه", "محاذير", "تحذيرات", "موازنة", "توازن",
        "غير منصوص", "لا توجد",  # Explicit acknowledgment of limits
    ]
    
    found = 0
    text_lower = answer_ar.lower()
    
    for marker in BOUNDARY_MARKERS:
        if marker in text_lower:
            found += 1
    
    # Score based on finding expected boundaries
    if boundaries_expected == 0:
        return 1.0  # No boundaries expected, full score
    
    score = min(1.0, found / boundaries_expected)
    
    return round(score, 3)


def compute_quote_budget_compliance_score(
    answer_ar: str,
    max_quote_length: int = 200,
    max_total_quotes: int = 15,
) -> float:
    """Compute score for quote budget compliance.
    
    Checks that quotes in the answer are within reasonable length limits.
    
    Args:
        answer_ar: Arabic answer text
        max_quote_length: Maximum length for individual quotes
        max_total_quotes: Maximum number of quotes
        
    Returns:
        Score in [0, 1] range (1.0 = fully compliant)
    """
    if not answer_ar:
        return 1.0  # No answer, nothing to violate
    
    # Count potential quotes (text between quotation marks)
    quote_markers = ['"', '"', '"', '«', '»', '"']
    
    in_quote = False
    current_quote_len = 0
    quotes_found = 0
    violations = 0
    
    for char in answer_ar:
        if char in quote_markers:
            if in_quote:
                # End of quote
                if current_quote_len > max_quote_length:
                    violations += 1
                quotes_found += 1
                current_quote_len = 0
            in_quote = not in_quote
        elif in_quote:
            current_quote_len += 1
    
    # Check total quotes
    if quotes_found > max_total_quotes:
        violations += (quotes_found - max_total_quotes) * 0.5
    
    # Score: 1.0 for no violations, decreases with violations
    if quotes_found == 0:
        return 1.0
    
    compliance = max(0.0, 1.0 - (violations / quotes_found))
    
    return round(compliance, 3)


def compute_pillar_coverage_score(
    pillars_covered: list[str],
    pillars_expected: int = 5,
) -> float:
    """Compute score for pillar coverage.
    
    Args:
        pillars_covered: List of pillar IDs covered in answer
        pillars_expected: Expected number of pillars (usually 5)
        
    Returns:
        Score in [0, 1] range
    """
    if pillars_expected == 0:
        return 1.0
    
    unique_pillars = len(set(pillars_covered))
    score = unique_pillars / pillars_expected
    
    return round(min(1.0, score), 3)


def score_world_model_answer(
    *,
    answer_ar: str,
    mechanism_trace: dict[str, Any],
    question_entities: list[dict[str, Any]],
    question_pillars: list[str],
    edges_available: int = 150,
    loops_detected: int = 20,
) -> dict[str, float]:
    """Score a world model synthesis answer.
    
    Combines all world model metrics into a single result dict.
    
    Args:
        answer_ar: Arabic answer text
        mechanism_trace: MechanismTrace dict with loops/interventions
        question_entities: Entities from question
        question_pillars: Pillars from question
        edges_available: Total mechanism edges in graph
        loops_detected: Total loops detected in graph
        
    Returns:
        Dict with all score components
    """
    # Extract data from mechanism_trace
    loops_used = mechanism_trace.get("loops", [])
    interventions = mechanism_trace.get("interventions", [])
    simulation_summary = mechanism_trace.get("simulation_summary", [])
    
    # Count edges used (from loops)
    edge_ids_used: set[str] = set()
    for loop in loops_used:
        for eid in loop.get("edge_ids", []):
            edge_ids_used.add(eid)
    
    # Get covered pillars
    pillars_covered = list(mechanism_trace.get("covered_pillars", []))
    
    # Compute all scores
    scores = {
        "loop_relevance": compute_loop_relevance_score(
            loops_used, question_entities, question_pillars
        ),
        "intervention_completeness": (
            compute_intervention_completeness_score(interventions[0])
            if interventions else 0.0
        ),
        "mechanism_coverage": compute_mechanism_coverage_score(
            edges_used=len(edge_ids_used),
            edges_available=edges_available,
            loops_used=len(loops_used),
            loops_detected=loops_detected,
        ),
        "boundary_completeness": compute_boundary_completeness_score(answer_ar),
        "quote_budget_compliance": compute_quote_budget_compliance_score(answer_ar),
        "pillar_coverage": compute_pillar_coverage_score(pillars_covered),
        
        # Raw counts for reference
        "loops_count": len(loops_used),
        "interventions_count": len(interventions),
        "simulations_count": len(simulation_summary),
        "edges_used_count": len(edge_ids_used),
        "pillars_covered_count": len(set(pillars_covered)),
    }
    
    # Compute overall score (weighted average of key metrics)
    overall = (
        0.25 * scores["loop_relevance"] +
        0.25 * scores["intervention_completeness"] +
        0.15 * scores["mechanism_coverage"] +
        0.15 * scores["boundary_completeness"] +
        0.1 * scores["quote_budget_compliance"] +
        0.1 * scores["pillar_coverage"]
    )
    scores["overall"] = round(overall, 3)
    
    return scores
