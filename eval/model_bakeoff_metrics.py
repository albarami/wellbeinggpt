"""Metrics computation for model bakeoff.

6 Dimensions:
A) Grounded accuracy (citations, integrity)
B) Depth (rubric, claim density, boundaries)
C) Connections (edges, pillars, chains)
D) Intent satisfaction (PASS_FULL, entities)
E) Naturalness (quote budget, redundancy, style)
F) Speed (latency, tokens)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from eval.types import EvalOutputRow


@dataclass
class BakeoffMetrics:
    """All metrics for one model/dataset/mode combination."""

    # A) Grounded Accuracy
    citation_validity_errors: int = 0
    unsupported_must_cite_count: int = 0
    unsupported_must_cite_rate: float = 0.0
    integrity_validator_hits: int = 0
    total_citations: int = 0

    # B) Depth
    rubric_score: float = 0.0  # 0-10
    claim_density_per_1k_chars: float = 0.0
    boundary_completeness: float = 0.0  # Rate of boundary markers present
    avg_word_count: float = 0.0

    # C) Connections
    used_edges_count: int = 0
    distinct_pillars: int = 0
    edge_type_diversity: int = 0  # Number of distinct relation types
    argument_chains_count: int = 0
    argument_chains_complete_rate: float = 0.0

    # D) Intent Satisfaction
    pass_full_rate: float = 0.0
    pass_partial_rate: float = 0.0
    required_entities_coverage: float = 0.0
    abstention_accuracy: float = 0.0

    # E) Naturalness
    quote_budget_compliance: float = 0.0  # Rate of quotes <= 25 words
    redundancy_rate: float = 0.0  # Duplicate sentence rate
    bullet_spam_rate: float = 0.0  # Excessive bullet points
    paragraph_flow_score: float = 0.0  # 0-1

    # F) Speed
    latency_p50_ms: float = 0.0
    latency_p95_ms: float = 0.0
    avg_latency_ms: float = 0.0
    total_tokens_in: int = 0
    total_tokens_out: int = 0

    # Counts
    total_questions: int = 0
    answered: int = 0
    abstained: int = 0


@dataclass
class DimensionScores:
    """Normalized scores per dimension (0-100 scale)."""

    grounding: float = 0.0  # A
    depth: float = 0.0  # B
    connections: float = 0.0  # C
    intent: float = 0.0  # D
    naturalness: float = 0.0  # E
    speed: float = 0.0  # F

    weighted_total: float = 0.0  # Final combined score


def compute_weighted_score(m: BakeoffMetrics) -> DimensionScores:
    """Compute dimension scores and weighted total.

    Weights (from spec):
    - 40% Depth + Intent
    - 30% Connections
    - 20% Naturalness
    - 10% Speed
    """
    # A) Grounding (pass/fail gating, not scored here)
    grounding = 100.0
    if m.citation_validity_errors > 0:
        grounding = 0.0
    elif m.unsupported_must_cite_rate > 0.1:
        grounding = max(0, 100 - m.unsupported_must_cite_rate * 500)

    # B) Depth (rubric 0-10 -> 0-100, + claim density bonus + boundary bonus)
    depth = m.rubric_score * 10  # 0-100
    depth += min(m.claim_density_per_1k_chars * 5, 20)  # Up to 20 bonus
    depth += m.boundary_completeness * 10  # Up to 10 bonus
    depth = min(100, depth)

    # C) Connections
    # edges (up to 50 pts), pillars (up to 25 pts), diversity (up to 15 pts), chains (up to 10 pts)
    conn_edges = min(m.used_edges_count * 5, 50)
    conn_pillars = min(m.distinct_pillars * 5, 25)
    conn_div = min(m.edge_type_diversity * 5, 15)
    conn_chains = m.argument_chains_complete_rate * 10
    connections = conn_edges + conn_pillars + conn_div + conn_chains

    # D) Intent satisfaction
    intent = m.pass_full_rate * 60 + m.pass_partial_rate * 20 + m.required_entities_coverage * 20

    # E) Naturalness
    nat = m.quote_budget_compliance * 40
    nat += (1 - m.redundancy_rate) * 20
    nat += (1 - m.bullet_spam_rate) * 20
    nat += m.paragraph_flow_score * 20
    naturalness = min(100, nat)

    # F) Speed (faster is better, baseline 5000ms p95 = 100)
    if m.latency_p95_ms > 0:
        speed = max(0, 100 - (m.latency_p95_ms - 2000) / 80)
    else:
        speed = 50  # Unknown

    # Weighted total:
    # 40% (Depth 20% + Intent 20%)
    # 30% Connections
    # 20% Naturalness
    # 10% Speed
    weighted = (
        depth * 0.20
        + intent * 0.20
        + connections * 0.30
        + naturalness * 0.20
        + speed * 0.10
    )

    return DimensionScores(
        grounding=grounding,
        depth=depth,
        connections=connections,
        intent=intent,
        naturalness=naturalness,
        speed=speed,
        weighted_total=round(weighted, 2),
    )


def _count_words(text: str) -> int:
    return len((text or "").split())


def _extract_sentences(text: str) -> list[str]:
    """Extract sentences from Arabic text."""
    if not text:
        return []
    # Split on Arabic/English sentence terminators
    parts = re.split(r'[.،؟!؛\n]', text)
    return [p.strip() for p in parts if p.strip() and len(p.strip()) > 10]


def _normalize_sentence(s: str) -> str:
    """Normalize for duplicate detection."""
    return re.sub(r'\s+', ' ', s.strip().lower())


def _has_boundary_marker(text: str) -> bool:
    """Check for boundary/limitation markers."""
    markers = ["حدود", "غير منصوص", "لم يرد", "خارج نطاق", "لا يتضمن الإطار"]
    return any(m in (text or "") for m in markers)


def _count_bullets(text: str) -> int:
    """Count bullet points."""
    lines = (text or "").splitlines()
    return sum(1 for ln in lines if ln.strip().startswith(("-", "*", "•", "١", "٢", "٣")))


def _paragraph_flow_score(text: str) -> float:
    """Heuristic for paragraph flow (vs. pure list spam)."""
    if not text:
        return 0.0
    lines = text.splitlines()
    if len(lines) < 3:
        return 1.0

    bullets = _count_bullets(text)
    prose_lines = sum(1 for ln in lines if ln.strip() and not ln.strip().startswith(("-", "*", "•")))

    if bullets == 0:
        return 1.0
    ratio = prose_lines / max(1, bullets)
    return min(1.0, ratio / 2)


def compute_all_metrics(
    outputs: list[EvalOutputRow],
    dataset_by_id: dict[str, dict[str, Any]],
    *,
    integrity_hits: int = 0,
) -> BakeoffMetrics:
    """Compute all bakeoff metrics from evaluation outputs."""
    m = BakeoffMetrics()
    m.total_questions = len(outputs)
    m.integrity_validator_hits = integrity_hits

    latencies: list[float] = []
    all_sentences: list[str] = []
    rubric_scores: list[float] = []
    word_counts: list[int] = []
    edge_types: set[str] = set()
    pillars_seen: set[str] = set()

    pass_full = 0
    pass_partial = 0
    has_boundary = 0
    total_claims = 0
    unsupported_claims = 0
    quote_compliant = 0
    total_quotes = 0
    bullets_total = 0
    flow_scores: list[float] = []

    for r in outputs:
        d = dataset_by_id.get(r.id, {})

        if r.abstained:
            m.abstained += 1
        else:
            m.answered += 1

        # Latency
        if r.latency_ms > 0:
            latencies.append(float(r.latency_ms))

        # Citations
        m.total_citations += len(r.citations)
        for c in r.citations:
            total_quotes += 1
            if _count_words(c.quote) <= 25:
                quote_compliant += 1

        # Claims
        for cl in r.claims:
            if cl.requires_evidence and cl.support_policy.value == "must_cite":
                total_claims += 1
                if not cl.evidence.supporting_spans:
                    unsupported_claims += 1

        # Rubric (simple heuristic if not pre-computed)
        ans = r.answer_ar or ""
        word_counts.append(_count_words(ans))

        # Simple rubric approximation
        score = 0
        if len(ans) > 100:
            score += 2
        if len(r.citations) >= 2:
            score += 2
        if r.graph_trace.edges or r.graph_trace.paths:
            score += 2
        if _has_boundary_marker(ans):
            score += 2
            has_boundary += 1
        if r.graph_trace.argument_chains:
            score += 2
        rubric_scores.append(min(10, score))

        # Graph connections
        m.used_edges_count += len(r.graph_trace.used_edges)
        m.argument_chains_count += len(r.graph_trace.argument_chains)

        for ue in r.graph_trace.used_edges:
            edge_types.add(ue.relation_type)
            # Extract pillar from node names (heuristic)
            for node in [ue.from_node, ue.to_node]:
                if "pillar:" in node:
                    pillars_seen.add(node.split(":")[-1])

        # Naturalness
        sentences = _extract_sentences(ans)
        all_sentences.extend(sentences)
        bullets_total += _count_bullets(ans)
        flow_scores.append(_paragraph_flow_score(ans))

        # Intent satisfaction (from dataset expectations)
        expect_abstain = d.get("expect_abstain", False)
        if expect_abstain and r.abstained:
            pass_full += 1
        elif not expect_abstain and not r.abstained and len(r.citations) > 0:
            if r.graph_trace.used_edges or r.graph_trace.edges:
                pass_full += 1
            else:
                pass_partial += 1

    # Aggregate metrics
    if latencies:
        latencies.sort()
        m.avg_latency_ms = sum(latencies) / len(latencies)
        m.latency_p50_ms = latencies[len(latencies) // 2]
        m.latency_p95_ms = latencies[int(len(latencies) * 0.95)] if len(latencies) > 1 else latencies[0]

    if rubric_scores:
        m.rubric_score = sum(rubric_scores) / len(rubric_scores)

    if word_counts:
        m.avg_word_count = sum(word_counts) / len(word_counts)
        total_chars = sum(word_counts) * 5  # Rough char estimate
        if total_chars > 0:
            m.claim_density_per_1k_chars = (total_claims / total_chars) * 1000

    if m.answered > 0:
        m.boundary_completeness = has_boundary / m.answered

    if total_claims > 0:
        m.unsupported_must_cite_count = unsupported_claims
        m.unsupported_must_cite_rate = unsupported_claims / total_claims

    if m.total_questions > 0:
        m.pass_full_rate = pass_full / m.total_questions
        m.pass_partial_rate = pass_partial / m.total_questions

    if total_quotes > 0:
        m.quote_budget_compliance = quote_compliant / total_quotes

    # Redundancy
    if all_sentences:
        normalized = [_normalize_sentence(s) for s in all_sentences]
        unique = len(set(normalized))
        m.redundancy_rate = 1 - (unique / len(normalized))

    # Bullet spam
    if m.answered > 0:
        avg_bullets = bullets_total / m.answered
        m.bullet_spam_rate = min(1.0, avg_bullets / 20)  # >20 bullets = 100% spam

    if flow_scores:
        m.paragraph_flow_score = sum(flow_scores) / len(flow_scores)

    m.distinct_pillars = len(pillars_seen)
    m.edge_type_diversity = len(edge_types)

    if m.used_edges_count > 0:
        m.argument_chains_complete_rate = min(1.0, m.argument_chains_count / m.used_edges_count)

    return m
