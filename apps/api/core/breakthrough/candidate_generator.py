"""Deterministic candidate generation for breakthrough mode.

Generates K grounded candidates using ONLY:
- SCHOLAR_LINK edges from DB with justification_spans
- Deterministic graph traversal (sorted edge IDs)
- NO LLM reasoning

Hard requirements met:
- K candidates from grounded edges only
- Deterministic graph traversal (sorted edge IDs, fixed seed)
- All orderings are deterministic (sorted)
"""

from __future__ import annotations

from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text

from apps.api.core.breakthrough.candidate_ranker import Candidate


async def fetch_scholar_link_edges_sorted(
    session: AsyncSession,
    pillar_ids: list[str] | None = None,
    max_edges: int = 50,
) -> list[dict[str, Any]]:
    """
    Fetch SCHOLAR_LINK edges with justification spans, sorted by edge ID for determinism.
    
    All returned edges have at least one justification span (grounded).
    """
    # Build query with optional pillar filter
    base_query = """
        SELECT DISTINCT
            e.id::text as edge_id,
            e.from_type,
            e.from_id,
            e.to_type,
            e.to_id,
            e.relation_type,
            ejs.chunk_id,
            ejs.span_start,
            ejs.span_end,
            ejs.quote
        FROM edge e
        JOIN edge_justification_span ejs ON e.id = ejs.edge_id
        WHERE e.rel_type = 'SCHOLAR_LINK'
          AND e.status = 'approved'
          AND ejs.quote IS NOT NULL
          AND ejs.quote != ''
    """
    
    params: dict[str, Any] = {"max_edges": max_edges}
    
    if pillar_ids:
        base_query += """
          AND (
            (e.from_type = 'pillar' AND e.from_id = ANY(:pillar_ids))
            OR (e.to_type = 'pillar' AND e.to_id = ANY(:pillar_ids))
          )
        """
        params["pillar_ids"] = pillar_ids
    
    base_query += """
        ORDER BY e.id, ejs.span_start
        LIMIT :max_edges
    """
    
    try:
        result = await session.execute(text(base_query), params)
        rows = result.fetchall()
    except Exception:
        return []
    
    # Group by edge_id
    edges_by_id: dict[str, dict[str, Any]] = {}
    for row in rows:
        edge_id = str(row[0])
        if edge_id not in edges_by_id:
            edges_by_id[edge_id] = {
                "edge_id": edge_id,
                "from_node": f"{row[1]}:{row[2]}",
                "to_node": f"{row[3]}:{row[4]}",
                "relation_type": str(row[5] or ""),
                "justification_spans": [],
            }
        
        edges_by_id[edge_id]["justification_spans"].append({
            "chunk_id": str(row[6] or ""),
            "span_start": int(row[7] or 0),
            "span_end": int(row[8] or 0),
            "quote": str(row[9] or ""),
        })
    
    # Return sorted by edge_id for determinism
    return [edges_by_id[eid] for eid in sorted(edges_by_id.keys())]


def _generate_candidate_by_central_node(
    edges: list[dict[str, Any]],
    central_node: str,
    candidate_name: str,
    max_edges: int = 12,
) -> Candidate:
    """Generate a candidate network centered on a specific node."""
    # Filter edges involving the central node
    relevant = [
        e for e in edges
        if central_node in (e.get("from_node", ""), e.get("to_node", ""))
    ]
    
    # Sort by edge_id for determinism, take max_edges
    relevant = sorted(relevant, key=lambda x: x.get("edge_id", ""))[:max_edges]
    
    return Candidate(
        name=candidate_name,
        used_edges=relevant,
        argument_chains=[],  # Will be built by answer composer
    )


def _generate_candidate_by_relation_type(
    edges: list[dict[str, Any]],
    relation_type: str,
    candidate_name: str,
    max_edges: int = 12,
) -> Candidate:
    """Generate a candidate network filtered by relation type."""
    relevant = [
        e for e in edges
        if e.get("relation_type", "").upper() == relation_type.upper()
    ]
    
    # Sort by edge_id for determinism
    relevant = sorted(relevant, key=lambda x: x.get("edge_id", ""))[:max_edges]
    
    return Candidate(
        name=candidate_name,
        used_edges=relevant,
        argument_chains=[],
    )


def _generate_candidate_by_pillar_diversity(
    edges: list[dict[str, Any]],
    candidate_name: str,
    max_edges: int = 12,
) -> Candidate:
    """Generate a candidate that maximizes pillar diversity."""
    # Score edges by pillar diversity (prefer edges connecting different pillars)
    def edge_diversity_score(e: dict[str, Any]) -> tuple[int, str]:
        fn = e.get("from_node", "")
        tn = e.get("to_node", "")
        
        # Cross-pillar edges score higher
        if fn.startswith("pillar:") and tn.startswith("pillar:") and fn != tn:
            return (2, e.get("edge_id", ""))
        # Value-level edges score medium
        if "core_value:" in fn or "core_value:" in tn:
            return (1, e.get("edge_id", ""))
        # Same-pillar edges score lower
        return (0, e.get("edge_id", ""))
    
    # Sort by diversity score DESC, then edge_id ASC for determinism
    sorted_edges = sorted(edges, key=lambda e: (-edge_diversity_score(e)[0], edge_diversity_score(e)[1]))
    
    return Candidate(
        name=candidate_name,
        used_edges=sorted_edges[:max_edges],
        argument_chains=[],
    )


def _generate_candidate_all_edges(
    edges: list[dict[str, Any]],
    candidate_name: str,
    max_edges: int = 12,
) -> Candidate:
    """Generate a baseline candidate using first N edges (sorted by ID)."""
    sorted_edges = sorted(edges, key=lambda x: x.get("edge_id", ""))[:max_edges]
    
    return Candidate(
        name=candidate_name,
        used_edges=sorted_edges,
        argument_chains=[],
    )


async def generate_candidates(
    session: AsyncSession,
    question: str,
    intent: str,
    detected_pillar_ids: list[str] | None = None,
    max_candidates: int = 5,
    max_edges_per_candidate: int = 12,
) -> list[Candidate]:
    """
    Generate K grounded candidates using ONLY:
    - SCHOLAR_LINK edges from DB with justification_spans
    - Deterministic graph traversal (sorted edge IDs)
    - NO LLM reasoning
    
    Args:
        session: Database session
        question: The question text (for context, not LLM)
        intent: Detected intent type (global_synthesis, network_build, etc.)
        detected_pillar_ids: Pillar IDs from entity resolution
        max_candidates: Maximum candidates to generate (K)
        max_edges_per_candidate: Max edges per candidate network
    
    Returns:
        List of Candidate objects, all grounded and deterministic
    """
    # Fetch all relevant edges (sorted by ID for determinism)
    edges = await fetch_scholar_link_edges_sorted(
        session,
        pillar_ids=detected_pillar_ids,
        max_edges=max_edges_per_candidate * max_candidates * 2,  # Fetch extra for filtering
    )
    
    if not edges:
        return []
    
    candidates: list[Candidate] = []
    
    # Candidate 1: All edges (baseline)
    candidates.append(_generate_candidate_all_edges(
        edges, "cand_baseline", max_edges_per_candidate
    ))
    
    # Candidate 2: Pillar diversity (maximize cross-pillar coverage)
    candidates.append(_generate_candidate_by_pillar_diversity(
        edges, "cand_diverse", max_edges_per_candidate
    ))
    
    # Candidate 3-4: By relation type (ENABLES, REINFORCES)
    for i, rel_type in enumerate(["ENABLES", "REINFORCES"]):
        cand = _generate_candidate_by_relation_type(
            edges, rel_type, f"cand_rel_{rel_type.lower()}", max_edges_per_candidate
        )
        if cand.used_edges:
            candidates.append(cand)
    
    # Candidate 5+: By detected pillar (if available)
    if detected_pillar_ids:
        for i, pid in enumerate(sorted(detected_pillar_ids)[:2]):  # Max 2 pillar-centric
            cand = _generate_candidate_by_central_node(
                edges, f"pillar:{pid}", f"cand_pillar_{pid}", max_edges_per_candidate
            )
            if cand.used_edges:
                candidates.append(cand)
    
    # Dedupe by edge set (in case strategies produce identical candidates)
    seen_edge_sets: set[frozenset[str]] = set()
    unique_candidates: list[Candidate] = []
    
    for c in candidates:
        edge_ids = frozenset(e.get("edge_id", "") for e in c.used_edges)
        if edge_ids and edge_ids not in seen_edge_sets:
            seen_edge_sets.add(edge_ids)
            unique_candidates.append(c)
    
    # Return up to max_candidates
    return unique_candidates[:max_candidates]


def generate_candidates_sync(
    edges: list[dict[str, Any]],
    intent: str,
    detected_pillar_ids: list[str] | None = None,
    max_candidates: int = 5,
    max_edges_per_candidate: int = 12,
) -> list[Candidate]:
    """
    Synchronous candidate generation from pre-fetched edges.
    
    Use this when edges are already loaded (e.g., from semantic_edges_fallback).
    """
    if not edges:
        return []
    
    candidates: list[Candidate] = []
    
    # Same generation strategies as async version
    candidates.append(_generate_candidate_all_edges(
        edges, "cand_baseline", max_edges_per_candidate
    ))
    
    candidates.append(_generate_candidate_by_pillar_diversity(
        edges, "cand_diverse", max_edges_per_candidate
    ))
    
    for rel_type in ["ENABLES", "REINFORCES"]:
        cand = _generate_candidate_by_relation_type(
            edges, rel_type, f"cand_rel_{rel_type.lower()}", max_edges_per_candidate
        )
        if cand.used_edges:
            candidates.append(cand)
    
    if detected_pillar_ids:
        for pid in sorted(detected_pillar_ids)[:2]:
            cand = _generate_candidate_by_central_node(
                edges, f"pillar:{pid}", f"cand_pillar_{pid}", max_edges_per_candidate
            )
            if cand.used_edges:
                candidates.append(cand)
    
    # Dedupe
    seen: set[frozenset[str]] = set()
    unique: list[Candidate] = []
    for c in candidates:
        edge_ids = frozenset(e.get("edge_id", "") for e in c.used_edges)
        if edge_ids and edge_ids not in seen:
            seen.add(edge_ids)
            unique.append(c)
    
    return unique[:max_candidates]
