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


def audit_class_balance(traces: list[dict]) -> dict[str, Any]:
    """Audit class balance: positives vs negatives."""
    total_selected = 0
    total_rejected = 0
    selected_per_trace = []
    rejected_per_trace = []
    
    for trace in traces:
        candidates = trace.get("candidate_edges", [])
        selected = sum(1 for c in candidates if c.get("is_selected"))
        rejected = len(candidates) - selected
        
        total_selected += selected
        total_rejected += rejected
        selected_per_trace.append(selected)
        rejected_per_trace.append(rejected)
    
    total = total_selected + total_rejected
    positive_rate = total_selected / total if total > 0 else 0
    
    return {
        "total_candidates": total,
        "total_positives": total_selected,
        "total_negatives": total_rejected,
        "positive_rate": positive_rate,
        "mean_selected_per_trace": np.mean(selected_per_trace) if selected_per_trace else 0,
        "mean_rejected_per_trace": np.mean(rejected_per_trace) if rejected_per_trace else 0,
        "std_selected_per_trace": np.std(selected_per_trace) if selected_per_trace else 0,
        "status": "OK" if 0.05 <= positive_rate <= 0.40 else "WARNING",
        "message": f"Positive rate {positive_rate:.1%} is within expected range (5-40%)" 
                   if 0.05 <= positive_rate <= 0.40 
                   else f"Positive rate {positive_rate:.1%} is outside expected range",
    }


def audit_hardness(traces: list[dict]) -> dict[str, Any]:
    """Audit hardness: quality score distribution for positives vs negatives."""
    pos_scores = []
    neg_scores = []
    
    for trace in traces:
        for candidate in trace.get("candidate_edges", []):
            score = candidate.get("quality_score", 0.0)
            if candidate.get("is_selected"):
                pos_scores.append(score)
            else:
                neg_scores.append(score)
    
    if not pos_scores or not neg_scores:
        return {
            "status": "ERROR",
            "message": "Not enough positives or negatives to compute hardness",
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
        "status": status,
        "message": message,
    }


def audit_coverage(traces: list[dict]) -> dict[str, Any]:
    """Audit coverage: distribution of relation types and node types."""
    relation_types = defaultdict(int)
    from_types = defaultdict(int)
    to_types = defaultdict(int)
    intents = defaultdict(int)
    
    for trace in traces:
        intents[trace.get("intent", "unknown")] += 1
        for candidate in trace.get("candidate_edges", []):
            relation_types[candidate.get("relation_type", "unknown")] += 1
            from_types[candidate.get("from_type", "unknown")] += 1
            to_types[candidate.get("to_type", "unknown")] += 1
    
    # Check for dominance (one pattern > 70%)
    total_relations = sum(relation_types.values())
    dominant_relation = max(relation_types.values()) / total_relations if total_relations > 0 else 0
    
    total_from = sum(from_types.values())
    dominant_from = max(from_types.values()) / total_from if total_from > 0 else 0
    
    warnings = []
    if dominant_relation > 0.70:
        top_rel = max(relation_types, key=relation_types.get)
        warnings.append(f"Relation type '{top_rel}' dominates ({dominant_relation:.1%})")
    if dominant_from > 0.70:
        top_from = max(from_types, key=from_types.get)
        warnings.append(f"From type '{top_from}' dominates ({dominant_from:.1%})")
    
    return {
        "relation_type_distribution": dict(relation_types),
        "from_type_distribution": dict(from_types),
        "to_type_distribution": dict(to_types),
        "intent_distribution": dict(intents),
        "status": "WARNING" if warnings else "OK",
        "warnings": warnings,
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
    
    # 1. Class Balance
    logger.info("\n" + "-" * 50)
    logger.info("1. CLASS BALANCE")
    logger.info("-" * 50)
    balance = audit_class_balance(traces)
    logger.info(f"   Total candidates: {balance['total_candidates']:,}")
    logger.info(f"   Positives: {balance['total_positives']:,}")
    logger.info(f"   Negatives: {balance['total_negatives']:,}")
    logger.info(f"   Positive rate: {balance['positive_rate']:.1%}")
    logger.info(f"   Mean selected/trace: {balance['mean_selected_per_trace']:.1f}")
    logger.info(f"   Mean rejected/trace: {balance['mean_rejected_per_trace']:.1f}")
    logger.info(f"   Status: [{balance['status']}] {balance['message']}")
    
    # 2. Hardness
    logger.info("\n" + "-" * 50)
    logger.info("2. HARDNESS (Learning Signal)")
    logger.info("-" * 50)
    hardness = audit_hardness(traces)
    if "positive_mean_quality" in hardness:
        logger.info(f"   Positive mean quality: {hardness['positive_mean_quality']:.3f}")
        logger.info(f"   Negative mean quality: {hardness['negative_mean_quality']:.3f}")
        logger.info(f"   Separation score: {hardness['separation_score']:.2f}")
    logger.info(f"   Status: [{hardness['status']}] {hardness['message']}")
    
    # 3. Coverage
    logger.info("\n" + "-" * 50)
    logger.info("3. COVERAGE")
    logger.info("-" * 50)
    coverage = audit_coverage(traces)
    logger.info("   Relation types:")
    for rt, count in sorted(coverage['relation_type_distribution'].items(), key=lambda x: -x[1])[:5]:
        logger.info(f"      {rt}: {count}")
    logger.info("   From types:")
    for ft, count in sorted(coverage['from_type_distribution'].items(), key=lambda x: -x[1]):
        logger.info(f"      {ft}: {count}")
    logger.info("   Intents:")
    for intent, count in sorted(coverage['intent_distribution'].items(), key=lambda x: -x[1]):
        logger.info(f"      {intent}: {count}")
    if coverage['warnings']:
        for w in coverage['warnings']:
            logger.info(f"   WARNING: {w}")
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
    
    # Overall Summary
    logger.info("\n" + "=" * 70)
    logger.info("AUDIT SUMMARY")
    logger.info("=" * 70)
    
    all_ok = all(
        r.get("status") == "OK" 
        for r in [balance, hardness, coverage, leakage]
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
    
    logger.info("\n")


if __name__ == "__main__":
    run_full_audit()
