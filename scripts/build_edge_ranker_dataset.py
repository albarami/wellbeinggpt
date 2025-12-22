"""
Build Pairwise Edge Ranker Training Dataset.

Reads edge traces and creates pairwise ranking examples for training.
Filters to semantic edges only (excludes STRUCTURAL_SIBLING).

Usage:
    python scripts/build_edge_ranker_dataset.py [--dry-run] [--verbose]

Output:
    data/phase2/edge_ranker/train.jsonl
    data/phase2/edge_ranker/val.jsonl
    data/phase2/edge_ranker/dataset_report.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import random
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
TRACE_DIR = PROJECT_ROOT / "data" / "phase2" / "edge_traces" / "train"
OUTPUT_DIR = PROJECT_ROOT / "data" / "phase2" / "edge_ranker"

# Semantic relation types (for training)
SEMANTIC_RELATION_TYPES = {
    "ENABLES", "REINFORCES", "COMPLEMENTS", "CONDITIONAL_ON",
    "INHIBITS", "TENSION_WITH", "RESOLVES_WITH"
}

# Rare types to oversample (increase representation)
RARE_RELATION_TYPES = {
    "ENABLES", "INHIBITS", "CONDITIONAL_ON", "TENSION_WITH", "RESOLVES_WITH"
}

# Train/val split ratio
VAL_RATIO = 0.2

# Sampling parameters
NEGATIVES_PER_POSITIVE = (3, 10)  # min, max negatives per positive
COMPLEMENTS_CAP_RATIO = 0.50  # Cap COMPLEMENTS to 50% of semantic samples
RARE_OVERSAMPLE_FACTOR = 3  # Oversample rare types by this factor


@dataclass
class DatasetStats:
    """Statistics for the built dataset."""
    
    total_traces: int = 0
    semantic_traces: int = 0  # Traces with at least one semantic edge
    total_pairs: int = 0
    train_pairs: int = 0
    val_pairs: int = 0
    
    # Relation type distribution (in final dataset)
    relation_type_dist: dict[str, int] = field(default_factory=dict)
    
    # Class balance
    positive_count: int = 0
    negative_count: int = 0
    
    # Questions
    unique_questions: int = 0
    train_questions: int = 0
    val_questions: int = 0
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "total_traces": self.total_traces,
            "semantic_traces": self.semantic_traces,
            "total_pairs": self.total_pairs,
            "train_pairs": self.train_pairs,
            "val_pairs": self.val_pairs,
            "relation_type_distribution": self.relation_type_dist,
            "positive_count": self.positive_count,
            "negative_count": self.negative_count,
            "unique_questions": self.unique_questions,
            "train_questions": self.train_questions,
            "val_questions": self.val_questions,
            "positive_rate": self.positive_count / max(1, self.positive_count + self.negative_count),
        }


def question_hash(question: str) -> str:
    """Generate stable hash for question (for deterministic split)."""
    return hashlib.sha256(question.encode("utf-8")).hexdigest()[:16]


def is_train_question(question: str, val_ratio: float = VAL_RATIO) -> bool:
    """Determine if question belongs to train set (deterministic by hash)."""
    h = question_hash(question)
    # Use first 4 hex chars as a 16-bit number
    val = int(h[:4], 16) / 0xFFFF
    return val >= val_ratio


def load_traces() -> list[dict]:
    """Load all trace records from directory."""
    traces = []
    if not TRACE_DIR.exists():
        logger.warning(f"Trace directory not found: {TRACE_DIR}")
        return traces
    
    for trace_file in sorted(TRACE_DIR.glob("*.jsonl")):
        with open(trace_file, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    traces.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return traces


def filter_semantic_candidates(candidates: list[dict]) -> list[dict]:
    """Filter candidates to semantic relation types only."""
    return [
        c for c in candidates
        if str(c.get("relation_type", "")).upper() in SEMANTIC_RELATION_TYPES
    ]


def extract_edge_features(edge: dict) -> dict[str, Any]:
    """Extract features for an edge (for training input)."""
    return {
        "edge_id": edge.get("edge_id", ""),
        "from_type": edge.get("from_type", ""),
        "from_id": edge.get("from_id", ""),
        "from_label": edge.get("from_label", ""),
        "to_type": edge.get("to_type", ""),
        "to_id": edge.get("to_id", ""),
        "to_label": edge.get("to_label", ""),
        "relation_type": edge.get("relation_type", ""),
        "quality_score": edge.get("quality_score", 0.0),
        "baseline_rank_position": edge.get("baseline_rank_position", -1),
        "justification_snippet": _get_justification_snippet(edge),
    }


def _get_justification_snippet(edge: dict, max_len: int = 200) -> str:
    """Get a snippet of justification text for the edge."""
    spans = edge.get("justification_spans", [])
    if not spans:
        return ""
    
    # Get first span's quote
    first_span = spans[0] if isinstance(spans, list) else {}
    quote = first_span.get("quote", "") if isinstance(first_span, dict) else ""
    return quote[:max_len]


def create_pairwise_examples(
    trace: dict,
    oversample_rare: bool = True,
) -> list[dict]:
    """Create pairwise ranking examples from a single trace.
    
    For each positive (selected) edge, pair with multiple negative (rejected) edges.
    """
    question = trace.get("question", "")
    intent = trace.get("intent", "")
    candidates = trace.get("candidate_edges", [])
    
    # Filter to semantic only
    semantic_candidates = filter_semantic_candidates(candidates)
    if not semantic_candidates:
        return []
    
    # Split into positives and negatives
    positives = [c for c in semantic_candidates if c.get("is_selected")]
    negatives = [c for c in semantic_candidates if not c.get("is_selected")]
    
    if not positives or not negatives:
        return []
    
    examples = []
    
    for pos_edge in positives:
        pos_relation = str(pos_edge.get("relation_type", "")).upper()
        
        # Determine number of negatives to sample
        num_negatives = random.randint(*NEGATIVES_PER_POSITIVE)
        num_negatives = min(num_negatives, len(negatives))
        
        # Sample negatives
        sampled_negatives = random.sample(negatives, num_negatives)
        
        # Create examples
        for neg_edge in sampled_negatives:
            example = {
                "question": question,
                "intent": intent,
                "positive": extract_edge_features(pos_edge),
                "negative": extract_edge_features(neg_edge),
                "pos_relation_type": pos_relation,
                "neg_relation_type": str(neg_edge.get("relation_type", "")).upper(),
            }
            examples.append(example)
        
        # Oversample rare relation types
        if oversample_rare and pos_relation in RARE_RELATION_TYPES:
            # Add more examples for rare types
            for _ in range(RARE_OVERSAMPLE_FACTOR - 1):
                if len(negatives) > 0:
                    extra_negatives = random.sample(
                        negatives, 
                        min(num_negatives, len(negatives))
                    )
                    for neg_edge in extra_negatives:
                        example = {
                            "question": question,
                            "intent": intent,
                            "positive": extract_edge_features(pos_edge),
                            "negative": extract_edge_features(neg_edge),
                            "pos_relation_type": pos_relation,
                            "neg_relation_type": str(neg_edge.get("relation_type", "")).upper(),
                            "oversampled": True,
                        }
                        examples.append(example)
    
    return examples


def apply_complements_cap(examples: list[dict], cap_ratio: float = COMPLEMENTS_CAP_RATIO) -> list[dict]:
    """Cap COMPLEMENTS examples to a maximum ratio of the dataset.
    
    This prevents COMPLEMENTS from dominating training even after filtering.
    Only applies cap if there are enough non-COMPLEMENTS examples.
    """
    complements_examples = [e for e in examples if e.get("pos_relation_type") == "COMPLEMENTS"]
    other_examples = [e for e in examples if e.get("pos_relation_type") != "COMPLEMENTS"]
    
    original_complements = len(complements_examples)
    
    # Only apply cap if we have enough other examples
    if len(other_examples) >= 10:
        # Calculate target count for COMPLEMENTS based on other examples
        other_count = len(other_examples)
        max_complements = int(other_count * cap_ratio / (1 - cap_ratio))
        
        if len(complements_examples) > max_complements:
            # Randomly sample to cap
            random.shuffle(complements_examples)
            complements_examples = complements_examples[:max_complements]
            logger.info(f"  Capped COMPLEMENTS from {original_complements} to {max_complements}")
    else:
        # Not enough other examples - keep all (early collection phase)
        logger.info(f"  Skipping COMPLEMENTS cap (not enough other examples: {len(other_examples)})")
    
    return other_examples + complements_examples


def build_dataset(
    traces: list[dict],
    dry_run: bool = False,
    verbose: bool = False,
) -> DatasetStats:
    """Build the complete training dataset from traces."""
    stats = DatasetStats()
    stats.total_traces = len(traces)
    
    # Group traces by question for split
    by_question: dict[str, list[dict]] = defaultdict(list)
    for trace in traces:
        question = trace.get("question", "")
        by_question[question].append(trace)
    
    stats.unique_questions = len(by_question)
    
    # Process traces and create examples
    all_train_examples = []
    all_val_examples = []
    
    for question, question_traces in by_question.items():
        is_train = is_train_question(question)
        
        for trace in question_traces:
            # Check if trace has semantic edges
            candidates = trace.get("candidate_edges", [])
            semantic_candidates = filter_semantic_candidates(candidates)
            if semantic_candidates:
                stats.semantic_traces += 1
            
            # Create pairwise examples
            examples = create_pairwise_examples(trace)
            
            if examples:
                if is_train:
                    all_train_examples.extend(examples)
                else:
                    all_val_examples.extend(examples)
        
        if is_train:
            stats.train_questions += 1
        else:
            stats.val_questions += 1
    
    # Apply COMPLEMENTS cap to both sets
    logger.info("Applying COMPLEMENTS cap...")
    all_train_examples = apply_complements_cap(all_train_examples)
    all_val_examples = apply_complements_cap(all_val_examples)
    
    # Shuffle
    random.shuffle(all_train_examples)
    random.shuffle(all_val_examples)
    
    stats.train_pairs = len(all_train_examples)
    stats.val_pairs = len(all_val_examples)
    stats.total_pairs = stats.train_pairs + stats.val_pairs
    
    # Compute relation type distribution
    for example in all_train_examples + all_val_examples:
        rel = example.get("pos_relation_type", "unknown")
        stats.relation_type_dist[rel] = stats.relation_type_dist.get(rel, 0) + 1
    
    # Count positives and negatives (each example has 1 pos, 1 neg)
    stats.positive_count = stats.total_pairs
    stats.negative_count = stats.total_pairs
    
    # Write outputs
    if not dry_run:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        
        train_path = OUTPUT_DIR / "train.jsonl"
        val_path = OUTPUT_DIR / "val.jsonl"
        report_path = OUTPUT_DIR / "dataset_report.json"
        
        with open(train_path, "w", encoding="utf-8") as f:
            for example in all_train_examples:
                f.write(json.dumps(example, ensure_ascii=False) + "\n")
        
        with open(val_path, "w", encoding="utf-8") as f:
            for example in all_val_examples:
                f.write(json.dumps(example, ensure_ascii=False) + "\n")
        
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(stats.to_dict(), f, ensure_ascii=False, indent=2)
        
        logger.info(f"Written:")
        logger.info(f"  {train_path} ({stats.train_pairs:,} pairs)")
        logger.info(f"  {val_path} ({stats.val_pairs:,} pairs)")
        logger.info(f"  {report_path}")
    
    return stats


def main():
    parser = argparse.ArgumentParser(description="Build edge ranker training dataset")
    parser.add_argument("--dry-run", action="store_true", help="Don't write output files")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()
    
    random.seed(args.seed)
    
    logger.info("=" * 70)
    logger.info("BUILD EDGE RANKER TRAINING DATASET")
    logger.info("=" * 70)
    
    # Load traces
    logger.info(f"\nLoading traces from {TRACE_DIR}...")
    traces = load_traces()
    
    if not traces:
        logger.error("No traces found. Enable EDGE_TRACE_LOGGING and collect data first.")
        return
    
    logger.info(f"Loaded {len(traces):,} traces")
    
    # Build dataset
    logger.info("\nBuilding pairwise ranking dataset...")
    stats = build_dataset(traces, dry_run=args.dry_run, verbose=args.verbose)
    
    # Print summary
    logger.info("\n" + "-" * 50)
    logger.info("DATASET SUMMARY")
    logger.info("-" * 50)
    logger.info(f"Total traces: {stats.total_traces:,}")
    logger.info(f"Traces with semantic edges: {stats.semantic_traces:,}")
    logger.info(f"Unique questions: {stats.unique_questions:,}")
    logger.info(f"  Train questions: {stats.train_questions:,}")
    logger.info(f"  Val questions: {stats.val_questions:,}")
    logger.info(f"\nTotal pairwise examples: {stats.total_pairs:,}")
    logger.info(f"  Train: {stats.train_pairs:,}")
    logger.info(f"  Val: {stats.val_pairs:,}")
    
    logger.info(f"\nRelation type distribution (positive edges):")
    for rel, count in sorted(stats.relation_type_dist.items(), key=lambda x: -x[1]):
        pct = count / stats.total_pairs * 100 if stats.total_pairs > 0 else 0
        logger.info(f"  {rel}: {count:,} ({pct:.1f}%)")
    
    # Check targets
    logger.info("\n" + "-" * 50)
    logger.info("TARGET CHECK")
    logger.info("-" * 50)
    
    # Check COMPLEMENTS ratio
    complements_count = stats.relation_type_dist.get("COMPLEMENTS", 0)
    complements_ratio = complements_count / stats.total_pairs if stats.total_pairs > 0 else 0
    complements_ok = complements_ratio <= COMPLEMENTS_CAP_RATIO
    logger.info(f"COMPLEMENTS ratio: {complements_ratio:.1%} (target: <={COMPLEMENTS_CAP_RATIO:.0%}) {'OK' if complements_ok else 'NEEDS_DOWNSAMPLE'}")
    
    # Check minimum examples
    min_pairs = 1000  # Minimum for meaningful training
    pairs_ok = stats.total_pairs >= min_pairs
    logger.info(f"Total pairs: {stats.total_pairs:,} (minimum: {min_pairs:,}) {'OK' if pairs_ok else 'NEED_MORE_TRACES'}")
    
    if args.dry_run:
        logger.info("\n[DRY RUN - No files written]")
    
    logger.info("\n")


if __name__ == "__main__":
    main()
