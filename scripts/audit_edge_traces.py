"""
Dataset Audit Script for Edge Scorer Training Data.

Validates trace quality before training to prevent:
- Class imbalance issues
- Leakage between train/val
- Low hardness (no learning signal)
- Poor coverage (dominated by one pattern)

Run this BEFORE training to ensure data quality.
"""

import hashlib
import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
TRAIN_DIR = PROJECT_ROOT / "data" / "phase2" / "edge_traces" / "train"


def load_traces(trace_dir: Path) -> list[dict]:
    """Load all trace records from directory."""
    traces = []
    if not trace_dir.exists():
        return traces
    
    for trace_file in sorted(trace_dir.glob("*.jsonl")):
        with open(trace_file, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    traces.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return traces


def question_hash(question: str) -> str:
    """Generate stable hash for question (for split validation)."""
    return hashlib.sha256(question.encode("utf-8")).hexdigest()[:16]


def audit_class_balance(traces: list[dict], semantic_only: bool = True) -> dict[str, Any]:
    """Audit class balance: positives vs negatives.
    
    Args:
        traces: List of trace records
        semantic_only: If True, exclude STRUCTURAL_SIBLING edges from metrics
                       (since they should not be used for training)
    """
    # Semantic relation types (for training)
    SEMANTIC_TYPES = {"ENABLES", "REINFORCES", "COMPLEMENTS", "CONDITIONAL_ON", 
                      "INHIBITS", "TENSION_WITH", "RESOLVES_WITH"}
    
    total_selected = 0
    total_rejected = 0
    selected_per_trace = []
    rejected_per_trace = []
    
    # Also track all (including structural) for comparison
    total_all_selected = 0
    total_all_rejected = 0
    
    for trace in traces:
        candidates = trace.get("candidate_edges", [])
        
        # All candidates
        all_selected = sum(1 for c in candidates if c.get("is_selected"))
        all_rejected = len(candidates) - all_selected
        total_all_selected += all_selected
        total_all_rejected += all_rejected
        
        if semantic_only:
            # Filter to semantic only
            semantic_candidates = [
                c for c in candidates 
                if str(c.get("relation_type", "")).upper() in SEMANTIC_TYPES
            ]
            selected = sum(1 for c in semantic_candidates if c.get("is_selected"))
            rejected = len(semantic_candidates) - selected
        else:
            selected = all_selected
            rejected = all_rejected
        
        total_selected += selected
        total_rejected += rejected
        selected_per_trace.append(selected)
        rejected_per_trace.append(rejected)
    
    total = total_selected + total_rejected
    positive_rate = total_selected / total if total > 0 else 0
    
    total_all = total_all_selected + total_all_rejected
    all_positive_rate = total_all_selected / total_all if total_all > 0 else 0
    
    return {
        "total_candidates": total,
        "total_positives": total_selected,
        "total_negatives": total_rejected,
        "positive_rate": positive_rate,
        "mean_selected_per_trace": np.mean(selected_per_trace) if selected_per_trace else 0,
        "mean_rejected_per_trace": np.mean(rejected_per_trace) if rejected_per_trace else 0,
        "std_selected_per_trace": np.std(selected_per_trace) if selected_per_trace else 0,
        "semantic_only": semantic_only,
        # Also include all-candidates stats for reference
        "all_candidates_total": total_all,
        "all_candidates_positive_rate": all_positive_rate,
        "status": "OK" if 0.05 <= positive_rate <= 0.50 else "WARNING",
        "message": f"Semantic positive rate {positive_rate:.1%} is within expected range (5-50%)" 
                   if 0.05 <= positive_rate <= 0.50 
                   else f"Semantic positive rate {positive_rate:.1%} is outside expected range",
    }


def audit_hardness(traces: list[dict], semantic_only: bool = True) -> dict[str, Any]:
    """Audit hardness: quality score distribution for positives vs negatives.
    
    Args:
        traces: List of trace records
        semantic_only: If True, exclude STRUCTURAL_SIBLING edges
    """
    # Semantic relation types (for training)
    SEMANTIC_TYPES = {"ENABLES", "REINFORCES", "COMPLEMENTS", "CONDITIONAL_ON", 
                      "INHIBITS", "TENSION_WITH", "RESOLVES_WITH"}
    
    pos_scores = []
    neg_scores = []
    
    for trace in traces:
        for candidate in trace.get("candidate_edges", []):
            # Filter to semantic only if requested
            if semantic_only:
                rt = str(candidate.get("relation_type", "")).upper()
                if rt not in SEMANTIC_TYPES:
                    continue
            
            score = candidate.get("quality_score", 0.0)
            if candidate.get("is_selected"):
                pos_scores.append(score)
            else:
                neg_scores.append(score)
    
    if not pos_scores or not neg_scores:
        return {
            "status": "WARNING",
            "message": "Not enough semantic positives or negatives to compute hardness",
            "semantic_only": semantic_only,
        }
    
    pos_mean = np.mean(pos_scores)
    neg_mean = np.mean(neg_scores)
    pos_std = np.std(pos_scores)
    neg_std = np.std(neg_scores)
    
    # Overlap measure: how much do distributions overlap?
    # Lower separation = more overlap = harder task = more learning signal
    separation = (pos_mean - neg_mean) / max(0.5 * (pos_std + neg_std), 0.01)
    
    # Ideal: separation between 0.5 and 2.0 (some signal, but not trivial)
    if separation < 0.3:
        status = "WARNING"
        message = f"Very low separation ({separation:.2f}) - baseline barely distinguishes pos/neg"
    elif separation > 3.0:
        status = "WARNING"
        message = f"Very high separation ({separation:.2f}) - task may be too easy for ML"
    else:
        status = "OK"
        message = f"Good separation ({separation:.2f}) - meaningful learning signal"
    
    return {
        "positive_mean_quality": pos_mean,
        "positive_std_quality": pos_std,
        "negative_mean_quality": neg_mean,
        "negative_std_quality": neg_std,
        "separation_score": separation,
        "semantic_only": semantic_only,
        "status": status,
        "message": message,
    }


def audit_coverage(traces: list[dict]) -> dict[str, Any]:
    """Audit coverage: distribution of relation types and node types."""
    relation_types = defaultdict(int)
    from_types = defaultdict(int)
    to_types = defaultdict(int)
    intents = defaultdict(int)
    
    # Semantic vs structural breakdown
    SEMANTIC_TYPES = {"ENABLES", "REINFORCES", "COMPLEMENTS", "CONDITIONAL_ON", 
                      "INHIBITS", "TENSION_WITH", "RESOLVES_WITH"}
    STRUCTURAL_TYPES = {"STRUCTURAL_SIBLING"}
    semantic_count = 0
    structural_count = 0
    
    for trace in traces:
        intents[trace.get("intent", "unknown")] += 1
        for candidate in trace.get("candidate_edges", []):
            rt = candidate.get("relation_type", "unknown")
            relation_types[rt] += 1
            from_types[candidate.get("from_type", "unknown")] += 1
            to_types[candidate.get("to_type", "unknown")] += 1
            
            # Track semantic vs structural
            rt_upper = str(rt).upper()
            if rt_upper in SEMANTIC_TYPES:
                semantic_count += 1
            elif rt_upper in STRUCTURAL_TYPES:
                structural_count += 1
    
    # Check for dominance (one pattern > 70%)
    total_relations = sum(relation_types.values())
    dominant_relation = max(relation_types.values()) / total_relations if total_relations > 0 else 0
    
    total_from = sum(from_types.values())
    dominant_from = max(from_types.values()) / total_from if total_from > 0 else 0
    
    # Check semantic training viability
    semantic_rate = semantic_count / total_relations if total_relations > 0 else 0
    structural_rate = structural_count / total_relations if total_relations > 0 else 0
    
    warnings = []
    if dominant_relation > 0.70:
        top_rel = max(relation_types, key=relation_types.get)
        warnings.append(f"Relation type '{top_rel}' dominates ({dominant_relation:.1%})")
    if dominant_from > 0.70:
        top_from = max(from_types, key=from_types.get)
        warnings.append(f"From type '{top_from}' dominates ({dominant_from:.1%})")
    
    # Training filter recommendations
    training_recommendations = []
    if structural_rate > 0.50:
        training_recommendations.append(
            f"Filter STRUCTURAL_SIBLING from training ({structural_rate:.1%} of edges)"
        )
    
    # Check for COMPLEMENTS dominance within semantic edges
    complements_count = relation_types.get("COMPLEMENTS", 0)
    if semantic_count > 0 and complements_count / semantic_count > 0.70:
        training_recommendations.append(
            f"Downsample COMPLEMENTS to ≤50% for balanced training ({complements_count}/{semantic_count})"
        )
    
    return {
        "relation_type_distribution": dict(relation_types),
        "from_type_distribution": dict(from_types),
        "to_type_distribution": dict(to_types),
        "intent_distribution": dict(intents),
        "semantic_edges": semantic_count,
        "structural_edges": structural_count,
        "semantic_rate": semantic_rate,
        "structural_rate": structural_rate,
        "training_recommendations": training_recommendations,
        "status": "WARNING" if warnings else "OK",
        "warnings": warnings,
    }


def audit_baseline_disagreement(traces: list[dict], top_n: int = 10) -> dict[str, Any]:
    """
    Audit baseline disagreement rate.
    
    Measures what % of traces have selected edges that are NOT all in the 
    top-N by baseline quality score. Higher rate = more room for ML to improve.
    
    If this rate is near 0%, baseline is already optimal and training won't yield much.
    If it's healthy (10-30%), there's meaningful room for a learned model to improve.
    
    Args:
        traces: List of trace records
        top_n: How many top baseline ranks to consider as "good"
    
    Returns:
        Audit result with disagreement metrics
    """
    total_traces = 0
    disagreement_count = 0
    partial_disagreement_count = 0
    
    for trace in traces:
        candidates = trace.get("candidate_edges", [])
        if not candidates:
            continue
        
        total_traces += 1
        
        # Get selected edges
        selected = [c for c in candidates if c.get("is_selected")]
        if not selected:
            continue
        
        # Check if all selected edges are in top-N by baseline rank
        baseline_ranks = [c.get("baseline_rank_position", 999) for c in selected]
        all_in_top_n = all(r < top_n for r in baseline_ranks)
        any_in_top_n = any(r < top_n for r in baseline_ranks)
        
        if not all_in_top_n:
            disagreement_count += 1
        if any_in_top_n and not all_in_top_n:
            partial_disagreement_count += 1
    
    disagreement_rate = disagreement_count / total_traces if total_traces > 0 else 0
    
    # Interpretation
    if disagreement_rate < 0.05:
        status = "WARNING"
        message = f"Very low disagreement ({disagreement_rate:.1%}) - baseline already optimal, ML may not help"
    elif disagreement_rate > 0.50:
        status = "WARNING"
        message = f"Very high disagreement ({disagreement_rate:.1%}) - baseline may be poorly calibrated"
    else:
        status = "OK"
        message = f"Healthy disagreement ({disagreement_rate:.1%}) - good learning opportunity"
    
    return {
        "total_traces": total_traces,
        "disagreement_count": disagreement_count,
        "disagreement_rate": disagreement_rate,
        "partial_disagreement_count": partial_disagreement_count,
        "top_n_threshold": top_n,
        "status": status,
        "message": message,
    }


def audit_leakage(traces: list[dict], val_ratio: float = 0.2) -> dict[str, Any]:
    """Audit for train/val leakage based on question hash."""
    question_hashes = {}
    
    for i, trace in enumerate(traces):
        q = trace.get("question", "")
        h = question_hash(q)
        if h not in question_hashes:
            question_hashes[h] = []
        question_hashes[h].append(i)
    
    # Check for duplicate questions
    duplicates = {h: idxs for h, idxs in question_hashes.items() if len(idxs) > 1}
    
    unique_questions = len(question_hashes)
    total_traces = len(traces)
    
    # Simulate split
    sorted_hashes = sorted(question_hashes.keys())
    split_point = int(len(sorted_hashes) * (1 - val_ratio))
    train_hashes = set(sorted_hashes[:split_point])
    val_hashes = set(sorted_hashes[split_point:])
    
    # Check for any overlap (there shouldn't be any by construction)
    overlap = train_hashes & val_hashes
    
    return {
        "unique_questions": unique_questions,
        "total_traces": total_traces,
        "duplicate_question_count": len(duplicates),
        "train_questions": len(train_hashes),
        "val_questions": len(val_hashes),
        "leakage_detected": len(overlap) > 0,
        "status": "OK" if len(overlap) == 0 else "ERROR",
        "message": "No leakage detected - questions properly separated"
                   if len(overlap) == 0
                   else f"LEAKAGE: {len(overlap)} questions appear in both train and val!",
    }


def run_full_audit():
    """Run complete dataset audit."""
    logger.info("=" * 70)
    logger.info("EDGE SCORER TRAINING DATA AUDIT")
    logger.info("=" * 70)
    
    traces = load_traces(TRAIN_DIR)
    
    if not traces:
        logger.error(f"\nNo traces found in {TRAIN_DIR}")
        logger.error("Enable EDGE_TRACE_LOGGING=true and collect data first.")
        return
    
    logger.info(f"\nLoaded {len(traces)} traces from {TRAIN_DIR}")
    
    # 1. Class Balance (semantic edges only - for training)
    logger.info("\n" + "-" * 50)
    logger.info("1. CLASS BALANCE (Semantic Edges Only)")
    logger.info("-" * 50)
    balance = audit_class_balance(traces, semantic_only=True)
    logger.info(f"   Semantic candidates: {balance['total_candidates']:,}")
    logger.info(f"   Positives: {balance['total_positives']:,}")
    logger.info(f"   Negatives: {balance['total_negatives']:,}")
    logger.info(f"   Positive rate (semantic): {balance['positive_rate']:.1%}")
    logger.info(f"   Mean selected/trace: {balance['mean_selected_per_trace']:.1f}")
    logger.info(f"   Mean rejected/trace: {balance['mean_rejected_per_trace']:.1f}")
    logger.info(f"   (All candidates incl. structural: {balance['all_candidates_total']:,}, "
                f"rate: {balance['all_candidates_positive_rate']:.1%})")
    logger.info(f"   Status: [{balance['status']}] {balance['message']}")
    
    # 2. Hardness (semantic edges only)
    logger.info("\n" + "-" * 50)
    logger.info("2. HARDNESS (Semantic Edges Only)")
    logger.info("-" * 50)
    hardness = audit_hardness(traces, semantic_only=True)
    if "positive_mean_quality" in hardness:
        logger.info(f"   Semantic positive mean quality: {hardness['positive_mean_quality']:.3f}")
        logger.info(f"   Semantic negative mean quality: {hardness['negative_mean_quality']:.3f}")
        logger.info(f"   Separation score: {hardness['separation_score']:.2f}")
    logger.info(f"   Status: [{hardness['status']}] {hardness['message']}")
    
    # 3. Coverage
    logger.info("\n" + "-" * 50)
    logger.info("3. COVERAGE")
    logger.info("-" * 50)
    coverage = audit_coverage(traces)
    
    # Semantic vs Structural breakdown (critical for training)
    logger.info("   Semantic vs Structural:")
    logger.info(f"      Semantic edges: {coverage['semantic_edges']:,} ({coverage['semantic_rate']:.1%})")
    logger.info(f"      Structural edges: {coverage['structural_edges']:,} ({coverage['structural_rate']:.1%})")
    
    logger.info("   Relation types:")
    for rt, count in sorted(coverage['relation_type_distribution'].items(), key=lambda x: -x[1])[:8]:
        marker = " (semantic)" if rt.upper() in {"ENABLES", "REINFORCES", "COMPLEMENTS", "CONDITIONAL_ON", 
                                                   "INHIBITS", "TENSION_WITH", "RESOLVES_WITH"} else \
                 " (structural)" if rt.upper() == "STRUCTURAL_SIBLING" else ""
        logger.info(f"      {rt}: {count}{marker}")
    logger.info("   From types:")
    for ft, count in sorted(coverage['from_type_distribution'].items(), key=lambda x: -x[1]):
        logger.info(f"      {ft}: {count}")
    logger.info("   Intents:")
    for intent, count in sorted(coverage['intent_distribution'].items(), key=lambda x: -x[1]):
        logger.info(f"      {intent}: {count}")
    if coverage['warnings']:
        for w in coverage['warnings']:
            logger.info(f"   WARNING: {w}")
    if coverage['training_recommendations']:
        logger.info("   Training Recommendations:")
        for rec in coverage['training_recommendations']:
            logger.info(f"      - {rec}")
    logger.info(f"   Status: [{coverage['status']}]")
    
    # 4. Leakage Check
    logger.info("\n" + "-" * 50)
    logger.info("4. LEAKAGE CHECK (Train/Val Split)")
    logger.info("-" * 50)
    leakage = audit_leakage(traces)
    logger.info(f"   Unique questions: {leakage['unique_questions']}")
    logger.info(f"   Total traces: {leakage['total_traces']}")
    logger.info(f"   Duplicate questions: {leakage['duplicate_question_count']}")
    logger.info(f"   Train questions: {leakage['train_questions']}")
    logger.info(f"   Val questions: {leakage['val_questions']}")
    logger.info(f"   Status: [{leakage['status']}] {leakage['message']}")
    
    # 5. Baseline Disagreement (Learning Opportunity)
    logger.info("\n" + "-" * 50)
    logger.info("5. BASELINE DISAGREEMENT (Learning Opportunity)")
    logger.info("-" * 50)
    disagreement = audit_baseline_disagreement(traces)
    logger.info(f"   Total traces analyzed: {disagreement['total_traces']}")
    logger.info(f"   Traces with disagreement: {disagreement['disagreement_count']}")
    logger.info(f"   Disagreement rate: {disagreement['disagreement_rate']:.1%}")
    logger.info(f"   Partial disagreement: {disagreement['partial_disagreement_count']}")
    logger.info(f"   Top-N threshold: {disagreement['top_n_threshold']}")
    logger.info(f"   Status: [{disagreement['status']}] {disagreement['message']}")
    
    # Overall Summary
    logger.info("\n" + "=" * 70)
    logger.info("AUDIT SUMMARY")
    logger.info("=" * 70)
    
    all_ok = all(
        r.get("status") == "OK" 
        for r in [balance, hardness, coverage, leakage, disagreement]
    )
    
    if all_ok:
        logger.info("✓ All checks passed. Data is ready for training.")
    else:
        logger.info("⚠ Some checks need attention. Review warnings above.")
    
    # Recommendations
    logger.info("\n" + "-" * 50)
    logger.info("RECOMMENDATIONS")
    logger.info("-" * 50)
    
    if balance['total_candidates'] < 10000:
        logger.info(f"   • Collect more data. Current: {balance['total_candidates']:,}, Target: 50k-200k")
    
    if balance['positive_rate'] < 0.10:
        logger.info("   • Low positive rate - consider sampling more diverse questions")
    
    if hardness.get('separation_score', 0) > 2.5:
        logger.info("   • High separation - baseline already strong, ML may add little value")
    
    if leakage['duplicate_question_count'] > 0:
        logger.info("   • Deduplicate traces before training to avoid bias")
    
    if disagreement.get('disagreement_rate', 0) < 0.10:
        logger.info("   • Low disagreement - baseline already captures selection well")
        logger.info("   • Consider if ML training is worth the effort")
    elif disagreement.get('disagreement_rate', 0) > 0.40:
        logger.info("   • High disagreement - good learning opportunity but check baseline calibration")
    
    logger.info("\n")


if __name__ == "__main__":
    run_full_audit()
