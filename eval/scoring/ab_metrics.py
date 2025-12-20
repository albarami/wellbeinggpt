"""A/B Evaluation Metrics for Muḥāsibī value-add analysis.

Computes comparative metrics across evaluation modes to quantify:
- Accuracy improvements (unsupported claims reduction)
- Contract pass rate improvement
- Depth improvements (edges, chains, pillars)
- Integrity (quarantine effectiveness)
- Style metrics (quote count, paragraph count, reasoning leak rate)
- Value-add per millisecond latency

Enables attribution analysis:
- Integrity effect = RAG_ONLY_INTEGRITY - RAG_ONLY
- Muḥāsibī reasoning effect = FULL_SYSTEM - RAG_PLUS_GRAPH_INTEGRITY
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class AbSummary:
    """Summary metrics for a single evaluation mode."""
    
    mode: str
    n: int = 0
    
    # Accuracy metrics
    unsupported_must_cite_rate: float = 0.0
    citation_validity_errors: int = 0
    
    # Contract metrics
    contract_pass_full: float = 0.0
    contract_pass_partial: float = 0.0
    contract_fail: float = 0.0
    
    # Depth metrics
    mean_used_edges: float = 0.0
    mean_argument_chains: float = 0.0
    mean_distinct_pillars: float = 0.0
    
    # Integrity metrics
    quarantined_cites_blocked: int = 0
    
    # Style metrics
    mean_quote_count: float = 0.0
    mean_paragraph_count: float = 0.0
    reasoning_block_leak_rate: float = 0.0  # Should be 0 in user text
    
    # Latency metrics
    p50_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    mean_latency_ms: float = 0.0


@dataclass
class AbDelta:
    """Delta between two modes for a single metric."""
    
    metric: str
    baseline_mode: str
    full_mode: str
    baseline_value: float
    full_value: float
    delta: float
    delta_pct: float = 0.0
    delta_per_ms: float = 0.0  # Value-add per millisecond latency increase


@dataclass
class AbAttributionAnalysis:
    """Attribution analysis showing what contributes to value-add."""
    
    # Clean attribution deltas (as specified in requirements)
    integrity_effect: dict[str, AbDelta] = field(default_factory=dict)  # RAG_ONLY_INTEGRITY - RAG_ONLY
    integrity_graph_effect: dict[str, AbDelta] = field(default_factory=dict)  # RAG_PLUS_GRAPH_INTEGRITY - RAG_PLUS_GRAPH
    graph_effect: dict[str, AbDelta] = field(default_factory=dict)  # RAG_PLUS_GRAPH - RAG_ONLY
    muhasibi_effect: dict[str, AbDelta] = field(default_factory=dict)  # FULL - RAG_PLUS_GRAPH_INTEGRITY
    total_effect: dict[str, AbDelta] = field(default_factory=dict)  # FULL - RAG_ONLY
    full_vs_graph: dict[str, AbDelta] = field(default_factory=dict)  # FULL - RAG_PLUS_GRAPH


def _get_nested(d: dict[str, Any], path: str, default: Any = None) -> Any:
    """Get nested dict value by dot-separated path."""
    cur = d
    for p in path.split("."):
        if not isinstance(cur, dict) or p not in cur:
            return default
        cur = cur[p]
    return cur


def _count_quotes(text: str) -> int:
    """Count quoted text segments in Arabic answer."""
    # Count «...» quotes and "..." quotes
    guillemet_count = len(re.findall(r"«[^»]+»", text))
    quote_count = len(re.findall(r'"[^"]+"', text))
    # Count direct verse citations (common pattern in evidence)
    verse_count = len(re.findall(r"[﴿﴾]", text)) // 2
    return guillemet_count + quote_count + verse_count


def _count_paragraphs(text: str) -> int:
    """Count paragraphs (separated by double newlines or headers)."""
    if not text.strip():
        return 0
    # Split by double newlines or section headers
    parts = re.split(r"\n\n+|\n(?=[^\n-])", text)
    return len([p for p in parts if p.strip()])


def _has_reasoning_block_leak(text: str) -> bool:
    """Check if internal reasoning block leaked into user-facing text."""
    leak_markers = [
        "[[MUHASIBI_REASONING_START]]",
        "[[MUHASIBI_REASONING_END]]",
        "REASONING_START",
        "REASONING_END",
    ]
    return any(marker in text for marker in leak_markers)


def _distinct_pillars_from_edges(edges: list[dict[str, Any]]) -> int:
    """Count distinct pillars referenced in used edges."""
    pillar_ids: set[str] = set()
    for e in edges:
        fn = str(e.get("from_node") or "")
        tn = str(e.get("to_node") or "")
        if fn.startswith("pillar:"):
            pillar_ids.add(fn)
        if tn.startswith("pillar:"):
            pillar_ids.add(tn)
    return len(pillar_ids)


def load_jsonl(path: Path | str) -> list[dict[str, Any]]:
    """Load JSONL file into list of dicts."""
    rows: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def compute_mode_summary(mode: str, rows: list[dict[str, Any]]) -> AbSummary:
    """Compute summary metrics for a single mode's outputs."""
    n = len(rows)
    if n == 0:
        return AbSummary(mode=mode)
    
    # Collect raw metrics
    latencies = [r.get("latency_ms", 0) for r in rows]
    latencies_sorted = sorted(latencies)
    
    # Accuracy metrics
    unsupported_rates: list[float] = []
    citation_errors = 0
    
    # Contract metrics
    contracts = [_get_nested(r, "debug.contract_outcome", "") or "" for r in rows]
    pass_full = sum(1 for c in contracts if c == "PASS_FULL")
    pass_partial = sum(1 for c in contracts if c == "PASS_PARTIAL")
    fail = sum(1 for c in contracts if c in ("FAIL", "CONTRACT_UNMET"))
    
    # Depth metrics
    used_edges_counts: list[int] = []
    argument_chains_counts: list[int] = []
    distinct_pillars_counts: list[int] = []
    
    # Integrity metrics
    quarantined_total = 0
    
    # Style metrics
    quote_counts: list[int] = []
    paragraph_counts: list[int] = []
    reasoning_leaks = 0
    
    for r in rows:
        # Unsupported claims
        claims = r.get("claims", []) or []
        must_cite_claims = [c for c in claims if c.get("support_policy") == "must_cite"]
        unsupported = [
            c for c in must_cite_claims
            if not (c.get("evidence", {}).get("supporting_spans") or [])
        ]
        if must_cite_claims:
            unsupported_rates.append(len(unsupported) / len(must_cite_claims))
        else:
            unsupported_rates.append(0.0)
        
        # Citation validity (check for known bad patterns)
        citations = r.get("citations", []) or []
        for cit in citations:
            if not cit.get("source_id"):
                citation_errors += 1
        
        # Graph trace metrics
        graph_trace = r.get("graph_trace", {}) or {}
        used_edges = graph_trace.get("used_edges", []) or []
        argument_chains = graph_trace.get("argument_chains", []) or []
        
        used_edges_counts.append(len(used_edges))
        argument_chains_counts.append(len(argument_chains))
        distinct_pillars_counts.append(_distinct_pillars_from_edges(used_edges))
        
        # Integrity metrics
        debug = r.get("debug", {}) or {}
        quarantined_total += int(debug.get("quarantined_cites_blocked", 0))
        
        # Style metrics
        answer = r.get("answer_ar", "") or ""
        quote_counts.append(_count_quotes(answer))
        paragraph_counts.append(_count_paragraphs(answer))
        if _has_reasoning_block_leak(answer):
            reasoning_leaks += 1
    
    # Compute percentiles
    p50_idx = int(0.50 * (n - 1))
    p95_idx = int(0.95 * (n - 1))
    
    return AbSummary(
        mode=mode,
        n=n,
        unsupported_must_cite_rate=sum(unsupported_rates) / n if n else 0.0,
        citation_validity_errors=citation_errors,
        contract_pass_full=pass_full / n if n else 0.0,
        contract_pass_partial=pass_partial / n if n else 0.0,
        contract_fail=fail / n if n else 0.0,
        mean_used_edges=sum(used_edges_counts) / n if n else 0.0,
        mean_argument_chains=sum(argument_chains_counts) / n if n else 0.0,
        mean_distinct_pillars=sum(distinct_pillars_counts) / n if n else 0.0,
        quarantined_cites_blocked=quarantined_total,
        mean_quote_count=sum(quote_counts) / n if n else 0.0,
        mean_paragraph_count=sum(paragraph_counts) / n if n else 0.0,
        reasoning_block_leak_rate=reasoning_leaks / n if n else 0.0,
        p50_latency_ms=latencies_sorted[p50_idx] if n else 0.0,
        p95_latency_ms=latencies_sorted[p95_idx] if n else 0.0,
        mean_latency_ms=sum(latencies) / n if n else 0.0,
    )


def compute_delta(
    metric: str,
    baseline: AbSummary,
    full: AbSummary,
) -> AbDelta:
    """Compute delta for a specific metric between two modes."""
    baseline_val = getattr(baseline, metric, 0.0)
    full_val = getattr(full, metric, 0.0)
    delta = full_val - baseline_val
    
    # Compute percentage change
    if baseline_val != 0:
        delta_pct = (delta / abs(baseline_val)) * 100
    else:
        delta_pct = 100.0 if delta > 0 else 0.0
    
    # Compute value-add per millisecond
    latency_delta = full.mean_latency_ms - baseline.mean_latency_ms
    if latency_delta > 0:
        delta_per_ms = delta / latency_delta
    else:
        delta_per_ms = 0.0
    
    return AbDelta(
        metric=metric,
        baseline_mode=baseline.mode,
        full_mode=full.mode,
        baseline_value=baseline_val,
        full_value=full_val,
        delta=delta,
        delta_pct=delta_pct,
        delta_per_ms=delta_per_ms,
    )


def compute_all_deltas(baseline: AbSummary, full: AbSummary) -> dict[str, AbDelta]:
    """Compute deltas for all relevant metrics."""
    metrics = [
        "unsupported_must_cite_rate",
        "contract_pass_full",
        "mean_used_edges",
        "mean_argument_chains",
        "mean_distinct_pillars",
        "quarantined_cites_blocked",
        "mean_quote_count",
        "reasoning_block_leak_rate",
        "mean_latency_ms",
    ]
    return {m: compute_delta(m, baseline, full) for m in metrics}


def compute_attribution_analysis(summaries: dict[str, AbSummary]) -> AbAttributionAnalysis:
    """
    Compute attribution analysis showing what contributes to value-add.
    
    Computes the following clean attribution deltas:
    - FULL_SYSTEM minus RAG_ONLY (total effect)
    - FULL_SYSTEM minus RAG_PLUS_GRAPH (muhasibi+integrity over graph)
    - RAG_ONLY_INTEGRITY minus RAG_ONLY (pure integrity effect)
    - RAG_PLUS_GRAPH_INTEGRITY minus RAG_PLUS_GRAPH (integrity effect with graph)
    - FULL_SYSTEM minus RAG_PLUS_GRAPH_INTEGRITY (pure Muḥāsibī reasoning effect)
    
    Requires summaries for:
    - RAG_ONLY
    - RAG_ONLY_INTEGRITY
    - RAG_PLUS_GRAPH
    - RAG_PLUS_GRAPH_INTEGRITY
    - FULL_SYSTEM
    """
    analysis = AbAttributionAnalysis()
    
    rag_only = summaries.get("RAG_ONLY")
    rag_only_int = summaries.get("RAG_ONLY_INTEGRITY")
    rag_graph = summaries.get("RAG_PLUS_GRAPH")
    rag_graph_int = summaries.get("RAG_PLUS_GRAPH_INTEGRITY")
    full = summaries.get("FULL_SYSTEM")
    
    # Integrity effect: RAG_ONLY_INTEGRITY - RAG_ONLY
    if rag_only and rag_only_int:
        analysis.integrity_effect = compute_all_deltas(rag_only, rag_only_int)
    
    # Integrity + Graph effect: RAG_PLUS_GRAPH_INTEGRITY - RAG_PLUS_GRAPH
    if rag_graph and rag_graph_int:
        analysis.integrity_graph_effect = compute_all_deltas(rag_graph, rag_graph_int)
    
    # Graph effect: RAG_PLUS_GRAPH - RAG_ONLY
    if rag_only and rag_graph:
        analysis.graph_effect = compute_all_deltas(rag_only, rag_graph)
    
    # Muḥāsibī reasoning effect: FULL - RAG_PLUS_GRAPH_INTEGRITY
    if rag_graph_int and full:
        analysis.muhasibi_effect = compute_all_deltas(rag_graph_int, full)
    
    # Total effect: FULL - RAG_ONLY
    if rag_only and full:
        analysis.total_effect = compute_all_deltas(rag_only, full)
    
    # FULL vs Graph only: FULL - RAG_PLUS_GRAPH
    if rag_graph and full:
        analysis.full_vs_graph = compute_all_deltas(rag_graph, full)
    
    return analysis
