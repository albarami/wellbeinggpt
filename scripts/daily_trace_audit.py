"""
Daily Trace Collection Audit.

Checks trace collection progress and quality for edge scorer training.
Run daily to monitor collection status and decide when to start training.

Usage:
    python scripts/daily_trace_audit.py

Stop Condition (ready for training):
    - 2,000-5,000 PASS_FULL traces
    - ≥50k candidate rows (prefer 100k-200k)
    - Semantic positive rate: 5-50%
    - Hardness separation: 0.3-2.5
    - No single pattern >70%
"""

from __future__ import annotations

import json
import logging
import os
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
TRAIN_DIR = PROJECT_ROOT / "data" / "phase2" / "edge_traces" / "train"
DEBUG_DIR = PROJECT_ROOT / "data" / "phase2" / "edge_traces" / "debug"

# Semantic relation types
SEMANTIC_TYPES = {"ENABLES", "REINFORCES", "COMPLEMENTS", "CONDITIONAL_ON", 
                  "INHIBITS", "TENSION_WITH", "RESOLVES_WITH"}

# Stop condition thresholds
MIN_TRACES = 2000
MAX_TRACES = 5000
MIN_CANDIDATES = 50000
TARGET_CANDIDATES = 100000


def get_dir_size_mb(directory: Path) -> float:
    """Get directory size in MB."""
    if not directory.exists():
        return 0.0
    total = sum(f.stat().st_size for f in directory.glob("*.jsonl") if f.is_file())
    return total / (1024 * 1024)


def count_traces_and_candidates(trace_dir: Path) -> tuple[int, int, int, dict]:
    """Count traces, total candidates, and semantic candidates."""
    trace_count = 0
    total_candidates = 0
    semantic_candidates = 0
    relation_dist: dict[str, int] = defaultdict(int)
    
    if not trace_dir.exists():
        return 0, 0, 0, {}
    
    for trace_file in trace_dir.glob("*.jsonl"):
        with open(trace_file, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    trace = json.loads(line)
                    trace_count += 1
                    candidates = trace.get("candidate_edges", [])
                    total_candidates += len(candidates)
                    
                    for c in candidates:
                        rt = str(c.get("relation_type", "")).upper()
                        relation_dist[rt] += 1
                        if rt in SEMANTIC_TYPES:
                            semantic_candidates += 1
                except json.JSONDecodeError:
                    pass
    
    return trace_count, total_candidates, semantic_candidates, dict(relation_dist)


def main():
    # Load .env
    from eval.datasets.source_loader import load_dotenv_if_present
    load_dotenv_if_present()
    
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    logger.info("=" * 70)
    logger.info(f"DAILY TRACE COLLECTION AUDIT - {now}")
    logger.info("=" * 70)
    
    # Check environment
    edge_logging = os.environ.get("EDGE_TRACE_LOGGING", "false").lower() == "true"
    sample_rate = os.environ.get("EDGE_TRACE_SAMPLE_RATE", "0.25")
    
    logger.info(f"\nConfiguration:")
    logger.info(f"  EDGE_TRACE_LOGGING: {edge_logging}")
    logger.info(f"  EDGE_TRACE_SAMPLE_RATE: {sample_rate}")
    
    if not edge_logging:
        logger.warning("  ⚠ EDGE_TRACE_LOGGING is disabled. Enable to collect training data.")
    
    # Count training traces
    logger.info(f"\n" + "-" * 50)
    logger.info("TRAINING TRACES (PASS_FULL)")
    logger.info("-" * 50)
    
    train_traces, train_total, train_semantic, train_dist = count_traces_and_candidates(TRAIN_DIR)
    train_size = get_dir_size_mb(TRAIN_DIR)
    
    logger.info(f"  Trace count: {train_traces:,}")
    logger.info(f"  Total candidates: {train_total:,}")
    logger.info(f"  Semantic candidates: {train_semantic:,} ({train_semantic/max(1,train_total)*100:.1f}%)")
    logger.info(f"  Disk usage: {train_size:.2f} MB")
    
    # Count debug traces
    logger.info(f"\n" + "-" * 50)
    logger.info("DEBUG TRACES (PASS_PARTIAL/FAIL)")
    logger.info("-" * 50)
    
    debug_traces, debug_total, debug_semantic, debug_dist = count_traces_and_candidates(DEBUG_DIR)
    debug_size = get_dir_size_mb(DEBUG_DIR)
    
    logger.info(f"  Trace count: {debug_traces:,}")
    logger.info(f"  Disk usage: {debug_size:.2f} MB")
    
    # Relation type distribution
    logger.info(f"\n" + "-" * 50)
    logger.info("RELATION TYPE DISTRIBUTION (Training)")
    logger.info("-" * 50)
    
    if train_dist:
        for rt, count in sorted(train_dist.items(), key=lambda x: -x[1])[:10]:
            marker = " (semantic)" if rt in SEMANTIC_TYPES else " (structural)"
            pct = count / train_total * 100 if train_total > 0 else 0
            logger.info(f"  {rt}: {count:,} ({pct:.1f}%){marker}")
    else:
        logger.info("  No data yet")
    
    # Stop condition check
    logger.info(f"\n" + "-" * 50)
    logger.info("STOP CONDITION CHECK")
    logger.info("-" * 50)
    
    traces_ok = MIN_TRACES <= train_traces <= MAX_TRACES * 2
    candidates_ok = train_total >= MIN_CANDIDATES
    semantic_rate = train_semantic / max(1, train_total)
    semantic_ok = 0.05 <= semantic_rate <= 0.50
    
    logger.info(f"  Traces: {train_traces:,} (target: {MIN_TRACES:,}-{MAX_TRACES:,}) {'✓' if traces_ok else '✗'}")
    logger.info(f"  Candidates: {train_total:,} (minimum: {MIN_CANDIDATES:,}) {'✓' if candidates_ok else '✗'}")
    logger.info(f"  Semantic rate: {semantic_rate:.1%} (target: 5-50%) {'✓' if semantic_ok else '✗'}")
    
    # Recommendation
    logger.info(f"\n" + "-" * 50)
    logger.info("RECOMMENDATION")
    logger.info("-" * 50)
    
    if train_traces >= MIN_TRACES and candidates_ok:
        logger.info("  ✓ READY FOR TRAINING")
        logger.info("  Run: python scripts/build_edge_ranker_dataset.py")
        logger.info("  Then: python scripts/audit_edge_traces.py (final validation)")
    elif train_traces == 0:
        logger.info("  ✗ NO TRACES COLLECTED")
        logger.info("  1. Set EDGE_TRACE_LOGGING=true in .env")
        logger.info("  2. Send requests to the API")
        logger.info("  3. Check that requests hit edge-scoring intents")
    else:
        remaining_traces = max(0, MIN_TRACES - train_traces)
        remaining_candidates = max(0, MIN_CANDIDATES - train_total)
        logger.info(f"  ✗ CONTINUE COLLECTING")
        logger.info(f"  Need ~{remaining_traces:,} more traces")
        logger.info(f"  Need ~{remaining_candidates:,} more candidate rows")
        
        # Estimate time at current rate
        if train_traces > 0:
            avg_candidates_per_trace = train_total / train_traces
            traces_for_candidates = remaining_candidates / max(1, avg_candidates_per_trace)
            traces_needed = max(remaining_traces, traces_for_candidates)
            logger.info(f"  At current avg ({avg_candidates_per_trace:.1f} candidates/trace):")
            logger.info(f"    Need ~{int(traces_needed):,} more traces total")
    
    # Daily summary for logging
    summary = {
        "timestamp": now,
        "train_traces": train_traces,
        "train_candidates": train_total,
        "train_semantic": train_semantic,
        "train_size_mb": round(train_size, 2),
        "debug_traces": debug_traces,
        "ready_for_training": train_traces >= MIN_TRACES and candidates_ok,
    }
    
    # Append to daily log
    log_path = PROJECT_ROOT / "data" / "phase2" / "collection_log.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(summary, ensure_ascii=False) + "\n")
    
    logger.info(f"\n  (Logged to {log_path})")
    logger.info("\n")


if __name__ == "__main__":
    main()
