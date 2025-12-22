"""
Edge Selection Module - Implements top-K selection with diversity constraints.

This module creates real training data by:
1. Building a candidate pool (N=50-100 edges)
2. Scoring and ranking edges deterministically
3. Selecting top-K (8-12) with diversity constraints
4. Logging both candidate pool and selected/rejected edges

The selection step creates real negatives for training:
- "High baseline rank but not selected" = hard negative
- "Lower baseline but selected" = surprising positive
"""

import logging
from typing import Any

logger = logging.getLogger(__name__)

# Selection parameters
DEFAULT_CANDIDATE_POOL_SIZE = 50  # N candidates to consider
DEFAULT_TOP_K = 10  # K edges to select
MAX_PILLAR_TO_PILLAR_EDGES = 3  # Cap on pillar→pillar edges
MIN_VALUE_LEVEL_EDGES = 2  # Prefer value-level edges
MIN_SEMANTIC_SELECTED = 1  # Guarantee at least 1 semantic edge if available (for training signal)


def compute_edge_score(edge: dict[str, Any]) -> float:
    """
    Compute deterministic quality score for an edge.
    
    Higher score = better edge for selection.
    
    Factors:
    - span_count: More grounding = better
    - quote quality: Non-empty quotes = better
    - relation_type: Semantic types preferred
    - entity level: Value-level > pillar-level
    """
    score = 0.0
    
    # Span count (up to 0.4)
    spans = edge.get("justification_spans", [])
    if isinstance(spans, list):
        span_count = len(spans)
        score += min(span_count * 0.12, 0.36)
        
        # Quote quality
        good_quotes = sum(1 for s in spans if isinstance(s, dict) and s.get("quote", "").strip())
        score += min(good_quotes * 0.08, 0.24)
    
    # Relation type priority (0.15 max)
    relation_type = str(edge.get("relation_type", "")).upper()
    
    # Semantic relation types get full bonus
    # STRUCTURAL_SIBLING gets lower priority (used for candidate diversity but not training)
    semantic_types = {"ENABLES", "REINFORCES", "COMPLEMENTS", "CONDITIONAL_ON", 
                      "INHIBITS", "TENSION_WITH", "RESOLVES_WITH"}
    if relation_type in semantic_types:
        score += 0.15
    elif relation_type == "STRUCTURAL_SIBLING":
        score += 0.05  # Lower priority than semantic edges
    
    # Entity level priority (0.25 max) - value-level edges preferred
    src_type = str(edge.get("source_type", "")).lower()
    neighbor_type = str(edge.get("neighbor_type", "")).lower()
    
    value_types = ("core_value", "sub_value")
    if src_type in value_types and neighbor_type in value_types:
        score += 0.25  # Both value-level: best
    elif src_type in value_types or neighbor_type in value_types:
        score += 0.15  # One value-level: good
    elif src_type == "pillar" and neighbor_type == "pillar":
        score += 0.0  # Pillar-to-pillar: no bonus
    else:
        score += 0.05  # Other combinations
    
    # Strength score bonus if available
    try:
        strength = float(edge.get("strength_score", 0.0))
        score += min(strength * 0.1, 0.1)
    except (ValueError, TypeError):
        pass
    
    return min(score, 1.0)


def _is_pillar_to_pillar(edge: dict[str, Any]) -> bool:
    """Check if edge is pillar→pillar."""
    src_type = str(edge.get("source_type", "")).lower()
    neighbor_type = str(edge.get("neighbor_type", "")).lower()
    return src_type == "pillar" and neighbor_type == "pillar"


def _is_value_level(edge: dict[str, Any]) -> bool:
    """Check if edge involves value-level entities."""
    src_type = str(edge.get("source_type", "")).lower()
    neighbor_type = str(edge.get("neighbor_type", "")).lower()
    value_types = ("core_value", "sub_value")
    return src_type in value_types or neighbor_type in value_types


def _is_structural(edge: dict[str, Any]) -> bool:
    """Check if edge is structural (not semantic).
    
    STRUCTURAL_SIBLING edges are hierarchical/sibling relationships
    that should be excluded from semantic edge scorer training.
    """
    relation_type = str(edge.get("relation_type", "")).upper()
    return relation_type == "STRUCTURAL_SIBLING"


def _is_semantic(edge: dict[str, Any]) -> bool:
    """Check if edge is a semantic relation (for training).
    
    Semantic edges have verb-marker-based relations:
    - ENABLES, REINFORCES, COMPLEMENTS, CONDITIONAL_ON
    - INHIBITS, TENSION_WITH, RESOLVES_WITH
    """
    relation_type = str(edge.get("relation_type", "")).upper()
    semantic_types = {"ENABLES", "REINFORCES", "COMPLEMENTS", "CONDITIONAL_ON", 
                      "INHIBITS", "TENSION_WITH", "RESOLVES_WITH"}
    return relation_type in semantic_types


# Relation types that are valid for training
SEMANTIC_RELATION_TYPES = {"ENABLES", "REINFORCES", "COMPLEMENTS", "CONDITIONAL_ON", 
                           "INHIBITS", "TENSION_WITH", "RESOLVES_WITH"}
STRUCTURAL_RELATION_TYPES = {"STRUCTURAL_SIBLING"}


def _edge_signature(edge: dict[str, Any]) -> str:
    """Generate unique signature for deduplication."""
    src = f"{edge.get('source_type', '')}:{edge.get('source_id', '')}"
    dst = f"{edge.get('neighbor_type', '')}:{edge.get('neighbor_id', '')}"
    rel = edge.get("relation_type", "")
    # Normalize direction
    if src > dst:
        src, dst = dst, src
    return f"{src}|{rel}|{dst}"


def select_top_k_edges(
    candidate_edges: list[dict[str, Any]],
    k: int = DEFAULT_TOP_K,
    max_pillar_to_pillar: int = MAX_PILLAR_TO_PILLAR_EDGES,
    min_value_level: int = MIN_VALUE_LEVEL_EDGES,
    min_semantic: int = MIN_SEMANTIC_SELECTED,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """
    Select top-K edges from candidate pool with diversity constraints.
    
    Args:
        candidate_edges: Full candidate pool (N edges)
        k: Number of edges to select
        max_pillar_to_pillar: Cap on pillar→pillar edges
        min_value_level: Minimum value-level edges to include
        min_semantic: Minimum semantic edges to include (for training signal)
    
    Returns:
        (selected_edges, rejected_edges)
    """
    if not candidate_edges:
        return [], []
    
    # Score all candidates
    scored = []
    for edge in candidate_edges:
        edge_copy = dict(edge)
        edge_copy["_selection_score"] = compute_edge_score(edge)
        edge_copy["_signature"] = _edge_signature(edge)
        edge_copy["_is_pillar_to_pillar"] = _is_pillar_to_pillar(edge)
        edge_copy["_is_value_level"] = _is_value_level(edge)
        edge_copy["_is_semantic"] = _is_semantic(edge)
        edge_copy["_is_structural"] = _is_structural(edge)
        scored.append(edge_copy)
    
    # Sort by score (descending)
    scored.sort(key=lambda x: -x["_selection_score"])
    
    # Selection with diversity constraints
    selected: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    seen_signatures: set[str] = set()
    pillar_to_pillar_count = 0
    value_level_count = 0
    
    # First pass: prioritize value-level edges to meet minimum
    for edge in scored:
        if len(selected) >= k:
            break
        
        sig = edge["_signature"]
        if sig in seen_signatures:
            rejected.append(edge)
            continue
        
        if edge["_is_value_level"]:
            selected.append(edge)
            seen_signatures.add(sig)
            value_level_count += 1
            if edge["_is_pillar_to_pillar"]:
                pillar_to_pillar_count += 1
    
    # Second pass: fill remaining slots with other edges
    for edge in scored:
        if len(selected) >= k:
            break
        
        sig = edge["_signature"]
        if sig in seen_signatures:
            continue
        
        # Check pillar-to-pillar cap
        if edge["_is_pillar_to_pillar"]:
            if pillar_to_pillar_count >= max_pillar_to_pillar:
                rejected.append(edge)
                continue
            pillar_to_pillar_count += 1
        
        selected.append(edge)
        seen_signatures.add(sig)
        if edge["_is_value_level"]:
            value_level_count += 1
    
    # Add remaining to rejected
    for edge in scored:
        sig = edge["_signature"]
        if sig not in seen_signatures:
            rejected.append(edge)
    
    # Count semantic vs structural
    semantic_count = sum(1 for e in selected if e.get("_is_semantic", False))
    structural_count = sum(1 for e in selected if e.get("_is_structural", False))
    
    # SEMANTIC GUARANTEE: If semantic candidates exist but not enough selected,
    # either add semantic edges (if room) or swap the lowest-scoring non-semantic edge.
    # This increases semantic positive rate for training without degrading answer quality.
    # NOTE: This only affects what select_top_k_edges returns, not what compose uses.
    # The actual training labels come from compose's used_edges, so this mainly helps
    # when compose does use the selected set (e.g., in graph-heavy intents).
    logger.info(f"SEMANTIC CHECK: semantic_count={semantic_count}, min_semantic={min_semantic}, selected={len(selected)}")
    if semantic_count < min_semantic:
        # Find semantic candidates in rejected pool
        semantic_rejected = [e for e in rejected if e.get("_is_semantic", False)]
        
        if semantic_rejected:
            # Sort semantic rejected by score (descending) to get the best one
            semantic_rejected.sort(key=lambda x: -x.get("_selection_score", 0))
            
            adds_needed = min_semantic - semantic_count
            
            # Case 1: If we have room (selected < k), just add semantic edges
            if len(selected) < k:
                adds_to_do = min(adds_needed, k - len(selected), len(semantic_rejected))
                for i in range(adds_to_do):
                    best_semantic = semantic_rejected[i]
                    rejected.remove(best_semantic)
                    selected.append(best_semantic)
                    semantic_count += 1
                    logger.debug(f"Semantic guarantee: added semantic edge (room available)")
            
            # Case 2: If full, swap non-semantic edges for semantic ones
            elif selected:
                # Find NON-semantic edges in selected (candidates for swap)
                non_semantic_selected = [e for e in selected if not e.get("_is_semantic", False)]
                
                if non_semantic_selected:
                    # Sort by score (ascending) to find the weakest one
                    non_semantic_selected.sort(key=lambda x: x.get("_selection_score", 0))
                    
                    # Perform swap: remove weakest non-semantic, add best semantic
                    swaps_needed = min(adds_needed, 
                                       len(semantic_rejected), 
                                       len(non_semantic_selected))
                    
                    for i in range(swaps_needed):
                        # Remove weakest non-semantic from selected
                        weak_edge = non_semantic_selected[i]
                        selected.remove(weak_edge)
                        rejected.append(weak_edge)
                        if weak_edge.get("_is_structural", False):
                            structural_count -= 1
                        
                        # Add best semantic from rejected
                        best_semantic = semantic_rejected[i]
                        rejected.remove(best_semantic)
                        selected.append(best_semantic)
                        semantic_count += 1
                        
                        logger.debug(
                            f"Semantic guarantee swap: removed non-semantic edge, added semantic edge"
                        )
    
    # Clean up internal fields
    for edge in selected + rejected:
        edge.pop("_selection_score", None)
        edge.pop("_signature", None)
        edge.pop("_is_pillar_to_pillar", None)
        edge.pop("_is_value_level", None)
        edge.pop("_is_semantic", None)
        edge.pop("_is_structural", None)
    
    logger.debug(
        f"Edge selection: {len(selected)} selected, {len(rejected)} rejected "
        f"(pillar-to-pillar: {pillar_to_pillar_count}, value-level: {value_level_count}, "
        f"semantic: {semantic_count}, structural: {structural_count})"
    )
    
    return selected, rejected


def apply_edge_selection_and_log(
    candidate_edges: list[dict[str, Any]],
    k: int = DEFAULT_TOP_K,
    intent: str = "",
    question: str = "",
    request_id: str = "",
    mode: str = "",
) -> list[dict[str, Any]]:
    """
    Apply edge selection and log for training data.
    
    This is the main entry point that:
    1. Selects top-K edges with diversity constraints
    2. Logs the candidate pool with selected/rejected labels
    3. Returns the selected edges for compose functions
    
    Args:
        candidate_edges: Full candidate pool
        k: Number to select
        intent: Intent type for logging
        question: Question text for logging
        request_id: Request ID for logging
        mode: Mode for logging
    
    Returns:
        Selected edges (top-K)
    """
    # Apply selection
    selected, rejected = select_top_k_edges(candidate_edges, k=k)
    
    # Log for training data collection
    try:
        from apps.api.graph.edge_candidate_logger import log_candidate_edges
        
        selected_ids = [str(e.get("edge_id", "")) for e in selected if e.get("edge_id")]
        
        log_candidate_edges(
            request_id=request_id,
            question=question,
            intent=intent,
            mode=mode,
            candidate_edges=candidate_edges,
            selected_edge_ids=selected_ids,
            contract_outcome="PASS_FULL",  # Will be updated by caller if needed
        )
    except Exception as e:
        logger.debug(f"Edge logging skipped: {e}")
    
    return selected
