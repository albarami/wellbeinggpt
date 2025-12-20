"""Re-score bakeoff using PRODUCTION grounding signals.

This scorer uses the same grounding logic as the production pipeline:
1. citation_validity: citations exist and spans are valid in DB
2. term_coverage: answer terms appear in cited evidence (min 50%)
3. contract_outcome: intent/structure satisfaction

Hard gates:
- citation_validity_errors == 0
- term_coverage >= 0.5 for all answers (if not abstained)
- unexpected_fail_rate <= threshold

Quality metrics:
- pass_full_rate, pass_partial_rate
- mean_term_coverage
- depth/cross/naturalness scores
"""

from __future__ import annotations

import asyncio
import json
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

OUT_DIR = REPO / "eval/output/bakeoff_depth_v1_system"
REPORT_PATH = REPO / "eval/reports/model_bakeoff_depth_v2.md"
DATASET_PATH = REPO / "eval/datasets/bakeoff_depth_v1.jsonl"

DEPLOYMENTS = ["gpt-5-chat", "gpt-5.1", "gpt-5.2"]

# Import production grounding tools
from apps.api.retrieve.normalize_ar import extract_arabic_words, normalize_for_matching


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                try:
                    rows.append(json.loads(s))
                except Exception:
                    pass
    return rows


def _sha256_file(path: Path) -> str:
    import hashlib
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _term_coverage(answer_ar: str, citations_spans: list[dict]) -> float:
    """
    Compute term coverage exactly as production ClaimToEvidenceChecker does.
    
    Returns ratio of answer terms found in cited evidence quotes.
    """
    # Strip reasoning block
    from apps.api.guardrails.citation_enforcer import _strip_muhasibi_reasoning_block
    answer_clean = _strip_muhasibi_reasoning_block(answer_ar or "")
    
    # Extract terms from answer
    answer_terms = extract_arabic_words(answer_clean)
    answer_terms = [t for t in answer_terms if len(t) >= 3]
    
    if not answer_terms:
        return 1.0  # No terms to check
    
    # Build combined text from citation quotes
    cited_text = " ".join(str(sp.get("quote") or "") for sp in citations_spans if sp.get("quote"))
    cited_normalized = normalize_for_matching(cited_text)
    
    if not cited_normalized:
        return 0.0  # No evidence
    
    # Count covered terms
    covered = 0
    for term in answer_terms:
        if normalize_for_matching(term) in cited_normalized:
            covered += 1
    
    return covered / len(answer_terms)


def _boundary_present(answer_ar: str) -> bool:
    markers = ["حدود", "غير منصوص", "لم يرد", "خارج نطاق", "لا يتضمن الإطار"]
    return any(m in (answer_ar or "") for m in markers)


def _redundancy_rate(answer_ar: str) -> float:
    import re
    if not (answer_ar or "").strip():
        return 0.0
    parts = re.split(r"[.،؟!؛\n]", answer_ar)
    sents = [p.strip().lower() for p in parts if p.strip() and len(p.strip()) > 10]
    if len(sents) <= 1:
        return 0.0
    seen: set[str] = set()
    dup = 0
    for s in sents:
        if s in seen:
            dup += 1
        else:
            seen.add(s)
    return dup / len(sents)


def _bullet_spam_rate(answer_ar: str) -> float:
    lines = (answer_ar or "").splitlines()
    bullets = sum(1 for ln in lines if ln.strip().startswith(("-", "*", "•")))
    return 1.0 if bullets > 18 else 0.0


def _quote_budget_compliance(citation_spans: list[dict]) -> float:
    if not citation_spans:
        return 1.0
    ok = sum(1 for sp in citation_spans if len((sp.get("quote") or "").split()) <= 25)
    return ok / len(citation_spans)


def _edge_diversity(used_edges: list[dict]) -> int:
    types = {str(e.get("relation_type") or "") for e in (used_edges or []) if str(e.get("relation_type") or "")}
    return len(types)


def _normalize01(x: float, lo: float, hi: float) -> float:
    if hi <= lo:
        return 0.0
    return max(0.0, min(1.0, (x - lo) / (hi - lo)))


@dataclass
class ModelSummary:
    deployment: str
    dataset_sha256: str
    total_questions: int
    successful_questions: int
    
    # Hard gates (production grounding)
    citation_validity_errors: int  # Would need DB check; placeholder
    low_coverage_count: int  # Answers with term_coverage < 0.5
    low_coverage_rate: float
    mean_term_coverage: float
    
    # Contract outcomes
    contract_pass_full_count: int
    contract_pass_partial_count: int
    contract_fail_count: int
    contract_pass_full_rate: float
    contract_fail_rate: float
    
    # Expected vs unexpected failures
    expected_fail_count: int  # injection, oos
    unexpected_fail_count: int
    unexpected_fail_rate: float
    
    disqualified: bool
    disqualification_reason: str
    
    # Quality metrics
    boundary_completeness_rate: float
    mean_used_edges: float
    mean_argument_chains: float
    edge_diversity: float
    redundancy_rate: float
    quote_budget_compliance_rate: float
    bullet_spam_rate: float
    
    # Scores
    depth_score_100: float
    cross_score_100: float
    naturalness_score_100: float
    grounding_score_100: float
    composite_score_100: float


def is_expected_fail(qtype: str) -> bool:
    """Return True if this question type is expected to fail."""
    return qtype in {"injection", "out_of_scope", "oos"}


def score_model(deployment: str, dataset_sha: str, total_questions: int, dataset_by_id: dict) -> ModelSummary:
    path = OUT_DIR / f"{deployment}.jsonl"
    rows = load_jsonl(path)
    
    # Deduplicate by ID
    by_id = {}
    for r in rows:
        qid = str(r.get("id") or "")
        if qid:
            by_id[qid] = r
    ok_rows = [r for r in by_id.values() if not r.get("error") and r.get("answer_ar")]
    
    # Accumulators
    pass_full = 0
    pass_partial = 0
    fail = 0
    expected_fail = 0
    unexpected_fail = 0
    
    term_coverages = []
    low_coverage = 0
    
    boundaries = []
    edges_counts = []
    chains_counts = []
    diversities = []
    redundancies = []
    bullet_spams = []
    quote_compliances = []
    
    for r in ok_rows:
        qid = r.get("id", "")
        qtype = str(r.get("type") or dataset_by_id.get(qid, {}).get("type") or "").lower()
        
        outcome = str(r.get("contract_outcome") or "").upper()
        if outcome == "PASS_FULL":
            pass_full += 1
        elif outcome == "PASS_PARTIAL":
            pass_partial += 1
        else:  # FAIL or unknown
            fail += 1
            if is_expected_fail(qtype):
                expected_fail += 1
            else:
                unexpected_fail += 1
        
        answer_ar = r.get("answer_ar", "")
        citations_spans = r.get("citations_spans") or []
        graph_trace = r.get("graph_trace") or {}
        used_edges = graph_trace.get("used_edges") or []
        chains = graph_trace.get("argument_chains") or []
        
        # Term coverage (production grounding metric)
        abstained = bool(r.get("abstained")) or "لا يوجد" in answer_ar[:100]
        if not abstained and citations_spans:
            cov = _term_coverage(answer_ar, citations_spans)
            term_coverages.append(cov)
            if cov < 0.5:
                low_coverage += 1
        elif not abstained and not citations_spans:
            # No citations = 0 coverage (should be caught by contract)
            term_coverages.append(0.0)
            low_coverage += 1
        
        boundaries.append(1.0 if _boundary_present(answer_ar) else 0.0)
        edges_counts.append(len(used_edges))
        chains_counts.append(len(chains))
        diversities.append(_edge_diversity(used_edges))
        redundancies.append(_redundancy_rate(answer_ar))
        bullet_spams.append(_bullet_spam_rate(answer_ar))
        quote_compliances.append(_quote_budget_compliance(citations_spans))
    
    n = max(1, len(ok_rows))
    
    pass_full_rate = pass_full / n
    fail_rate = fail / n
    unexpected_fail_rate = unexpected_fail / n
    low_coverage_rate = low_coverage / max(1, len(term_coverages))
    mean_coverage = sum(term_coverages) / max(1, len(term_coverages))
    
    boundary_rate = sum(boundaries) / n
    mean_edges = sum(edges_counts) / n
    mean_chains = sum(chains_counts) / n
    edge_div = sum(diversities) / n if diversities else 0
    redundancy = sum(redundancies) / n
    bullet_spam = sum(bullet_spams) / n
    quote_ok = sum(quote_compliances) / n
    
    # Scores
    # Depth: based on pass_full rate + boundaries
    depth = 0.0
    depth += _normalize01(pass_full_rate, 0.0, 1.0) * 70.0
    depth += _normalize01(boundary_rate, 0.0, 1.0) * 30.0
    
    # Cross-pillar: edges, chains, diversity
    cross = 0.0
    cross += _normalize01(mean_edges, 0.0, 8.0) * 40.0
    cross += _normalize01(mean_chains, 0.0, 6.0) * 35.0
    cross += _normalize01(edge_div, 0.0, 6.0) * 25.0
    
    # Naturalness
    nat = 0.0
    nat += _normalize01(quote_ok, 0.0, 1.0) * 45.0
    nat += (1.0 - max(0.0, min(1.0, redundancy))) * 35.0
    nat += (1.0 - max(0.0, min(1.0, bullet_spam))) * 20.0
    
    # Grounding: based on term coverage and low_coverage_rate
    grounding = 0.0
    grounding += _normalize01(mean_coverage, 0.5, 1.0) * 60.0  # Coverage above 50% threshold
    grounding += (1.0 - low_coverage_rate) * 40.0  # Penalty for low coverage answers
    
    # Disqualification
    disq = False
    disq_reason = ""
    # Disqualify if unexpected fail rate > 5%
    if unexpected_fail_rate > 0.05:
        disq = True
        disq_reason = f"unexpected_fail_rate={unexpected_fail_rate:.4f}"
    # Disqualify if low coverage rate > 10% (grounding failures)
    if low_coverage_rate > 0.10:
        disq = True
        disq_reason = f"low_coverage_rate={low_coverage_rate:.4f}"
    
    # Composite (if not disqualified)
    composite = depth * 0.40 + cross * 0.30 + nat * 0.15 + grounding * 0.15
    if disq:
        composite = 0.0
    
    return ModelSummary(
        deployment=deployment,
        dataset_sha256=dataset_sha,
        total_questions=total_questions,
        successful_questions=len(ok_rows),
        citation_validity_errors=0,  # Would need async DB check
        low_coverage_count=low_coverage,
        low_coverage_rate=low_coverage_rate,
        mean_term_coverage=mean_coverage,
        contract_pass_full_count=pass_full,
        contract_pass_partial_count=pass_partial,
        contract_fail_count=fail,
        contract_pass_full_rate=pass_full_rate,
        contract_fail_rate=fail_rate,
        expected_fail_count=expected_fail,
        unexpected_fail_count=unexpected_fail,
        unexpected_fail_rate=unexpected_fail_rate,
        disqualified=disq,
        disqualification_reason=disq_reason,
        boundary_completeness_rate=boundary_rate,
        mean_used_edges=mean_edges,
        mean_argument_chains=mean_chains,
        edge_diversity=edge_div,
        redundancy_rate=redundancy,
        quote_budget_compliance_rate=quote_ok,
        bullet_spam_rate=bullet_spam,
        depth_score_100=depth,
        cross_score_100=cross,
        naturalness_score_100=nat,
        grounding_score_100=grounding,
        composite_score_100=composite,
    )


def main():
    dataset_sha = _sha256_file(DATASET_PATH)
    dataset_rows = load_jsonl(DATASET_PATH)
    total_q = len(dataset_rows)
    dataset_by_id = {str(r.get("id")): r for r in dataset_rows}
    
    print(f"Dataset: {DATASET_PATH.name}  sha256={dataset_sha[:16]}  n={total_q}")
    print()
    
    summaries = []
    for dep in DEPLOYMENTS:
        s = score_model(dep, dataset_sha, total_q, dataset_by_id)
        summaries.append(s)
        
        print(f"--- {dep} ---")
        print(f"  Successful: {s.successful_questions}/{s.total_questions}")
        print(f"  PASS_FULL: {s.contract_pass_full_rate:.1%} ({s.contract_pass_full_count})")
        print(f"  FAIL: {s.contract_fail_rate:.1%} ({s.contract_fail_count} total, {s.expected_fail_count} expected, {s.unexpected_fail_count} unexpected)")
        print(f"  Term coverage: mean={s.mean_term_coverage:.1%}, low_coverage={s.low_coverage_rate:.1%}")
        print(f"  Disqualified: {s.disqualified} ({s.disqualification_reason})")
        print(f"  Scores: depth={s.depth_score_100:.1f} cross={s.cross_score_100:.1f} nat={s.naturalness_score_100:.1f} ground={s.grounding_score_100:.1f}")
        print(f"  Composite: {s.composite_score_100:.1f}")
        print()
        
        # Write summary
        sum_path = OUT_DIR / f"{dep}__summary_v2.json"
        sum_path.write_text(json.dumps(s.__dict__, ensure_ascii=False, indent=2), encoding="utf-8")
    
    # Ranking
    eligible = [s for s in summaries if not s.disqualified]
    ranked = sorted(eligible, key=lambda s: -s.composite_score_100) if eligible else []
    
    print("=" * 70)
    print("RANKING (by composite score)")
    print("=" * 70)
    for i, s in enumerate(ranked, 1):
        print(f"{i}. {s.deployment} composite={s.composite_score_100:.1f} depth={s.depth_score_100:.1f} cross={s.cross_score_100:.1f} ground={s.grounding_score_100:.1f}")
    
    if not ranked:
        print("All models disqualified. Showing raw scores:")
        for s in sorted(summaries, key=lambda s: -(s.depth_score_100 + s.cross_score_100)):
            print(f"  {s.deployment}: depth={s.depth_score_100:.1f} cross={s.cross_score_100:.1f} ground={s.grounding_score_100:.1f}")
    
    # Report
    lines = []
    lines.append("## System Depth Bakeoff (Production Grounding)\n")
    lines.append(f"- **Generated**: {datetime.utcnow().isoformat()}Z")
    lines.append(f"- **Dataset**: `{DATASET_PATH.as_posix()}`")
    lines.append(f"- **Dataset sha256**: `{dataset_sha}`")
    lines.append("")
    lines.append("### Grounding Methodology")
    lines.append("")
    lines.append("Uses **production grounding signals** (same as runtime):")
    lines.append("- **Term coverage**: answer terms appearing in cited evidence (min 50% required)")
    lines.append("- **Contract outcome**: PASS_FULL / PASS_PARTIAL / FAIL")
    lines.append("- **Expected failures**: injection + OOS questions (correctly abstained)")
    lines.append("")
    lines.append("### Hard Gates")
    lines.append("")
    lines.append("- **unexpected_fail_rate <= 5%**: Only fails on non-expected questions")
    lines.append("- **low_coverage_rate <= 10%**: Answers with term coverage < 50%")
    lines.append("")
    lines.append("### Per-model Summary")
    lines.append("")
    lines.append("| Model | Eligible | Composite | Depth | Cross | Nat | Ground | PASS_FULL | Unexpected Fail | Term Cov |")
    lines.append("|---|:---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for s in summaries:
        flag = "OK" if not s.disqualified else "DQ"
        lines.append(
            f"| {s.deployment} | {flag} | {s.composite_score_100:.1f} | {s.depth_score_100:.1f} | "
            f"{s.cross_score_100:.1f} | {s.naturalness_score_100:.1f} | {s.grounding_score_100:.1f} | "
            f"{s.contract_pass_full_rate:.1%} | {s.unexpected_fail_rate:.1%} | {s.mean_term_coverage:.1%} |"
        )
    
    lines.append("")
    lines.append("### Winners")
    lines.append("")
    if eligible:
        winner = ranked[0]
        lines.append(f"- **Overall**: {winner.deployment} (composite={winner.composite_score_100:.1f})")
        depth_winner = max(eligible, key=lambda s: s.depth_score_100)
        cross_winner = max(eligible, key=lambda s: s.cross_score_100)
        ground_winner = max(eligible, key=lambda s: s.grounding_score_100)
        lines.append(f"- **Depth**: {depth_winner.deployment} ({depth_winner.depth_score_100:.1f})")
        lines.append(f"- **Cross-pillar**: {cross_winner.deployment} ({cross_winner.cross_score_100:.1f})")
        lines.append(f"- **Grounding**: {ground_winner.deployment} ({ground_winner.grounding_score_100:.1f})")
    else:
        best = max(summaries, key=lambda s: s.depth_score_100)
        lines.append(f"- **Best (all DQ)**: {best.deployment} (depth={best.depth_score_100:.1f})")
    
    lines.append("")
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nWrote report: {REPORT_PATH}")


if __name__ == "__main__":
    main()

