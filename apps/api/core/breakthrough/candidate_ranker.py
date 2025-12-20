"""Deterministic candidate ranking for breakthrough mode.

Scores and selects the best candidate network/path using:
- Deterministic weighted scoring
- Stable sort with explicit tie-breakers for reproducibility

Hard requirements met:
- K candidates from grounded edges only (SCHOLAR_LINK + justification spans)
- Stable sort on ties: (score DESC, evidence_span_count DESC, distinct_pillars DESC, candidate_name ASC)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Candidate:
    """A candidate network/path for reasoning.
    
    All fields are grounded (from DB edges with justification spans).
    """
    
    name: str  # Unique identifier for deterministic tie-breaking
    used_edges: list[dict[str, Any]] = field(default_factory=list)
    argument_chains: list[dict[str, Any]] = field(default_factory=list)
    
    # Computed metrics (for scoring)
    distinct_pillars: int = 0
    evidence_span_count: int = 0
    boundary_hits: int = 0
    inference_penalty: float = 0.0  # Higher = worse (multi-span entailment penalty)
    
    # Final score (computed by score_candidate)
    score: float = 0.0


# Scoring weights (deterministic constants)
WEIGHT_PILLARS = 3.0
WEIGHT_EVIDENCE_SPANS = 0.3
WEIGHT_BOUNDARY = 1.5
WEIGHT_INFERENCE_PENALTY = 2.0


def _compute_distinct_pillars(edges: list[dict[str, Any]]) -> int:
    """Count distinct pillars referenced in edges."""
    pillar_ids: set[str] = set()
    for e in edges:
        fn = str(e.get("from_node") or "")
        tn = str(e.get("to_node") or "")
        if fn.startswith("pillar:"):
            pillar_ids.add(fn)
        if tn.startswith("pillar:"):
            pillar_ids.add(tn)
    return len(pillar_ids)


def _compute_evidence_span_count(edges: list[dict[str, Any]]) -> int:
    """Count total justification spans across edges."""
    count = 0
    for e in edges:
        spans = e.get("justification_spans") or []
        count += len(spans)
    return count


def _compute_inference_penalty(edges: list[dict[str, Any]]) -> float:
    """
    Compute inference penalty for multi-span entailment.
    
    Direct quotes are preferred over multi-span entailment.
    Penalty increases when edges have 2+ justification spans.
    """
    penalty = 0.0
    for e in edges:
        spans = e.get("justification_spans") or []
        if len(spans) >= 2:
            penalty += 0.2 * (len(spans) - 1)
    return penalty


def _compute_boundary_hits(edges: list[dict[str, Any]]) -> int:
    """Count edges that include boundary spans."""
    hits = 0
    for e in edges:
        boundary = e.get("boundary_spans") or []
        if boundary:
            hits += 1
    return hits


def score_candidate(c: Candidate) -> Candidate:
    """
    Compute score for a candidate using deterministic weighted formula.
    
    Modifies candidate in-place and returns it.
    """
    # Recompute metrics if not set
    if c.distinct_pillars == 0 and c.used_edges:
        c.distinct_pillars = _compute_distinct_pillars(c.used_edges)
    if c.evidence_span_count == 0 and c.used_edges:
        c.evidence_span_count = _compute_evidence_span_count(c.used_edges)
    if c.boundary_hits == 0 and c.used_edges:
        c.boundary_hits = _compute_boundary_hits(c.used_edges)
    if c.inference_penalty == 0.0 and c.used_edges:
        c.inference_penalty = _compute_inference_penalty(c.used_edges)
    
    # Deterministic weighted scoring
    c.score = (
        WEIGHT_PILLARS * float(c.distinct_pillars)
        + WEIGHT_EVIDENCE_SPANS * float(c.evidence_span_count)
        + WEIGHT_BOUNDARY * float(c.boundary_hits)
        - WEIGHT_INFERENCE_PENALTY * float(c.inference_penalty)
    )
    
    return c


def pick_best(candidates: list[Candidate]) -> Candidate | None:
    """
    Select the best candidate using deterministic stable sort.
    
    Tie-breaking order (all deterministic):
    1. score DESC
    2. evidence_span_count DESC
    3. distinct_pillars DESC
    4. name ASC (lexicographic)
    
    Returns None if candidates list is empty.
    """
    if not candidates:
        return None
    
    # Score all candidates
    scored = [score_candidate(c) for c in candidates]
    
    # Stable sort with explicit tie-breakers
    scored.sort(key=lambda x: (
        -x.score,
        -x.evidence_span_count,
        -x.distinct_pillars,
        x.name,  # Lexicographic for final tie-break
    ))
    
    return scored[0]


def rank_candidates(candidates: list[Candidate]) -> list[Candidate]:
    """
    Rank all candidates from best to worst (deterministic).
    
    Returns sorted list with scores computed.
    """
    if not candidates:
        return []
    
    scored = [score_candidate(c) for c in candidates]
    scored.sort(key=lambda x: (
        -x.score,
        -x.evidence_span_count,
        -x.distinct_pillars,
        x.name,
    ))
    
    return scored
