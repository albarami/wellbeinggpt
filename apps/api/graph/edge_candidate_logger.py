"""
Edge Candidate Logger - Logs candidate edge pools for training data collection.

Purpose:
- Log per-query candidate edge sets (before selection)
- Track selected vs rejected edges
- Collect real training labels for edge scorer

Logging policy:
- PASS_FULL → training logs (data/phase2/edge_traces/train/*.jsonl)
- PASS_PARTIAL/FAIL → debug logs (data/phase2/edge_traces/debug/*.jsonl)

Safeguards:
- Sampling: Only logs a configurable % of eligible requests
- Caps: Max candidates per request, max file size with rotation
"""

import json
import logging
import os
import random
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Configuration
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
TRACE_DIR = PROJECT_ROOT / "data" / "phase2" / "edge_traces"
TRAIN_DIR = TRACE_DIR / "train"
DEBUG_DIR = TRACE_DIR / "debug"

# Safeguards (configurable via env vars)
SAMPLE_RATE = float(os.getenv("EDGE_TRACE_SAMPLE_RATE", "0.25"))  # 25% default
MAX_CANDIDATES_PER_REQUEST = int(os.getenv("EDGE_TRACE_MAX_CANDIDATES", "100"))
MAX_FILE_SIZE_MB = float(os.getenv("EDGE_TRACE_MAX_FILE_MB", "50"))  # Rotate at 50MB

# Intents that require edge scoring for training
# IMPORTANT: Only include intents where edges are REQUIRED for the answer
# Do NOT include 'generic' - it dilutes training data with non-edge traces
EDGE_SCORING_INTENTS = {
    # Cross-pillar and network intents (MUST use edges)
    "cross_pillar",
    "cross_pillar_path",
    "connect_across_pillars",
    "network",
    "network_build",
    "value_network",
    # Synthesis intents (should use edges)
    "global_synthesis",
    "world_model_synthesis",
    "mechanism_query",
    # Relation-specific intents
    "tension",
    "compare",
    # Additional standard names (for flexibility)
    "network_build",
    "world_model_synthesis",
    "pillar_relationship",
    # API-level intent names (from main classifier)
    "connect_across_pillars",
    "practical_guidance",
    "value_network",
    "mechanism_query",
    "compare_pillars",
}


def should_log_edges(intent: Optional[str], apply_sampling: bool = True) -> bool:
    """
    Determine if candidate edges should be logged for this intent.
    
    Args:
        intent: The detected intent type
        apply_sampling: If True, applies probabilistic sampling
    
    Returns:
        True if this request should be logged
    """
    env_val = os.getenv("EDGE_TRACE_LOGGING", "false").lower()
    if env_val not in {"1", "true", "yes"}:
        logger.debug(f"Edge logging disabled: EDGE_TRACE_LOGGING={env_val}")
        return False
    
    intent_lower = (intent or "").lower().strip()
    if intent_lower not in EDGE_SCORING_INTENTS:
        logger.debug(f"Edge logging skipped: intent '{intent_lower}' not in {EDGE_SCORING_INTENTS}")
        return False
    
    # Apply probabilistic sampling to avoid disk growth
    if apply_sampling and random.random() > SAMPLE_RATE:
        logger.debug(f"Edge logging skipped: sampling (rate={SAMPLE_RATE})")
        return False
    
    logger.debug(f"Edge logging enabled for intent '{intent_lower}'")
    return True


def _get_trace_file(output_dir: Path, prefix: str) -> Path:
    """
    Get trace file with rotation support.
    
    Rotates when file exceeds MAX_FILE_SIZE_MB.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    today = datetime.utcnow().strftime("%Y%m%d")
    
    # Find current segment
    segment = 0
    while True:
        filename = f"{prefix}_{today}_{segment:03d}.jsonl"
        filepath = output_dir / filename
        
        if not filepath.exists():
            return filepath
        
        # Check file size
        size_mb = filepath.stat().st_size / (1024 * 1024)
        if size_mb < MAX_FILE_SIZE_MB:
            return filepath
        
        segment += 1
        if segment > 999:  # Safety limit
            return filepath


def compute_edge_quality_score(edge: dict[str, Any]) -> float:
    """
    Compute deterministic edge quality score based on evidence features.
    
    This is Option B: a baseline that doesn't require ML.
    
    Features:
    - span_count: Number of justification spans (more = better)
    - has_quote: Whether there's a non-empty quote
    - source_diversity: TODO when we have source_ids
    - boundary_present: TODO when we have boundary fields
    
    Returns:
        Score in [0, 1] where higher is better
    """
    score = 0.0
    
    # Justification spans count (up to 0.5)
    spans = edge.get("justification_spans", [])
    if isinstance(spans, list):
        span_count = len(spans)
        # More spans = better, with diminishing returns
        score += min(span_count * 0.15, 0.45)
        
        # Quality of spans (non-empty quotes)
        good_quotes = sum(1 for s in spans if isinstance(s, dict) and s.get("quote", "").strip())
        score += min(good_quotes * 0.1, 0.3)
    
    # Relation type priority (semantic types are better)
    relation_type = str(edge.get("relation_type", "")).upper()
    semantic_types = {"ENABLES", "REINFORCES", "COMPLEMENTS", "CONDITIONAL_ON", "INHIBITS", "TENSION_WITH", "RESOLVES_WITH"}
    if relation_type in semantic_types:
        score += 0.15
    
    # Entity type priority (values > pillars)
    src_type = str(edge.get("source_type", "")).lower()
    neighbor_type = str(edge.get("neighbor_type", "")).lower()
    if src_type in ("core_value", "sub_value") and neighbor_type in ("core_value", "sub_value"):
        score += 0.1
    elif src_type in ("core_value", "sub_value") or neighbor_type in ("core_value", "sub_value"):
        score += 0.05
    
    return min(score, 1.0)


def log_candidate_edges(
    *,
    request_id: str,
    question: str,
    intent: str,
    mode: str,
    candidate_edges: list[dict[str, Any]],
    selected_edge_ids: list[str],
    contract_outcome: str,
) -> None:
    """
    Log candidate edge pool for training data collection.
    
    Logging policy:
    - PASS_FULL → training logs (clean labels for ranking model)
    - PASS_PARTIAL/FAIL → debug logs (diagnosis, not for training)
    
    Safeguards:
    - Sampling: Only logs SAMPLE_RATE% of eligible requests
    - Caps: Max MAX_CANDIDATES_PER_REQUEST candidates per log
    - Rotation: New file when MAX_FILE_SIZE_MB exceeded
    
    Args:
        request_id: Unique request identifier
        question: The user question
        intent: Detected intent type
        mode: Request mode (answer, natural_chat, etc.)
        candidate_edges: All candidate edges (before selection)
        selected_edge_ids: IDs of edges that were selected (used_edges)
        contract_outcome: PASS_FULL, PASS_PARTIAL, or FAIL
    """
    logger.info(f"log_candidate_edges called: request_id={request_id[:8] if request_id else 'None'}, intent={intent}, outcome={contract_outcome}, candidates={len(candidate_edges)}")
    
    # Determine log type: training (PASS_FULL) or debug (others)
    is_training = contract_outcome == "PASS_FULL"
    
    # For training logs, apply sampling; for debug logs, always log (but check if enabled)
    if is_training:
        should_log = should_log_edges(intent, apply_sampling=True)
        logger.info(f"Training mode: should_log={should_log}")
        if not should_log:
            return
    else:
        # Debug logging: check if enabled but don't apply sampling
        should_log = should_log_edges(intent, apply_sampling=False)
        logger.info(f"Debug mode: should_log={should_log}")
        if not should_log:
            return
    
    try:
        # Choose output directory based on contract outcome
        output_dir = TRAIN_DIR if is_training else DEBUG_DIR
        prefix = "train" if is_training else "debug"
        
        # Apply candidate cap
        capped_candidates = candidate_edges[:MAX_CANDIDATES_PER_REQUEST]
        was_capped = len(candidate_edges) > MAX_CANDIDATES_PER_REQUEST
        
        # Compute quality scores and baseline ranks for all candidates
        enriched_candidates = []
        for edge in capped_candidates:
            edge_copy = edge.copy()
            edge_copy["quality_score"] = compute_edge_quality_score(edge)
            enriched_candidates.append(edge_copy)
        
        # Sort by baseline quality_score to get baseline rank positions
        # (This is how the deterministic baseline would rank them)
        sorted_by_baseline = sorted(
            enumerate(enriched_candidates),
            key=lambda x: -x[1]["quality_score"]  # Descending
        )
        baseline_rank_map = {idx: rank for rank, (idx, _) in enumerate(sorted_by_baseline)}
        
        # Add baseline_rank_position to each candidate
        for idx, edge in enumerate(enriched_candidates):
            edge["baseline_rank_position"] = baseline_rank_map[idx]
        
        selected_set = set(selected_edge_ids)
        
        # Compute final_selected_rank (position in the actual selected order)
        selected_rank_map = {eid: rank for rank, eid in enumerate(selected_edge_ids)}
        
        record = {
            "trace_id": str(uuid.uuid4()),
            "timestamp": datetime.utcnow().isoformat(),
            "request_id": request_id,
            "question": question,
            "intent": intent,
            "mode": mode,
            "contract_outcome": contract_outcome,
            "candidate_count": len(candidate_edges),
            "candidate_count_logged": len(capped_candidates),
            "was_capped": was_capped,
            "selected_count": len(selected_edge_ids),
            "candidate_edges": [
                {
                    "edge_id": str(e.get("edge_id", "")),
                    "from_type": str(e.get("source_type", "")),
                    "from_id": str(e.get("source_id", "")),
                    "to_type": str(e.get("neighbor_type", "")),
                    "to_id": str(e.get("neighbor_id", "")),
                    "relation_type": str(e.get("relation_type", "")),
                    "span_count": len(e.get("justification_spans", [])),
                    "quality_score": e.get("quality_score", 0.0),
                    "baseline_rank_position": e.get("baseline_rank_position", -1),
                    "final_selected_rank": selected_rank_map.get(str(e.get("edge_id", "")), -1),
                    "is_selected": str(e.get("edge_id", "")) in selected_set,
                }
                for e in enriched_candidates
            ],
            "selected_edge_ids": selected_edge_ids,
            "rejected_edge_ids": [
                str(e.get("edge_id", ""))
                for e in capped_candidates
                if str(e.get("edge_id", "")) not in selected_set
            ],
        }
        
        # Get trace file with rotation support
        trace_file = _get_trace_file(output_dir, prefix)
        
        with open(trace_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        
        log_type = "training" if is_training else "debug"
        logger.debug(f"Logged {len(capped_candidates)} candidate edges ({log_type}) for {intent}")
        
    except Exception as e:
        logger.warning(f"Failed to log candidate edges: {e}")


def get_edge_trace_stats() -> dict[str, Any]:
    """Get statistics about collected edge traces."""
    stats = {
        "training": {
            "total_traces": 0,
            "total_candidates": 0,
            "total_selected": 0,
            "total_rejected": 0,
            "traces_by_intent": {},
            "trace_files": [],
        },
        "debug": {
            "total_traces": 0,
            "total_candidates": 0,
            "total_selected": 0,
            "total_rejected": 0,
            "traces_by_intent": {},
            "trace_files": [],
        },
        "config": {
            "sample_rate": SAMPLE_RATE,
            "max_candidates_per_request": MAX_CANDIDATES_PER_REQUEST,
            "max_file_size_mb": MAX_FILE_SIZE_MB,
        },
    }
    
    for log_type, log_dir in [("training", TRAIN_DIR), ("debug", DEBUG_DIR)]:
        if not log_dir.exists():
            continue
        
        for trace_file in log_dir.glob("*.jsonl"):
            stats[log_type]["trace_files"].append(str(trace_file.name))
            try:
                with open(trace_file, "r", encoding="utf-8") as f:
                    for line in f:
                        try:
                            record = json.loads(line)
                            stats[log_type]["total_traces"] += 1
                            stats[log_type]["total_candidates"] += record.get("candidate_count_logged", 0)
                            stats[log_type]["total_selected"] += record.get("selected_count", 0)
                            stats[log_type]["total_rejected"] += (
                                record.get("candidate_count_logged", 0) - record.get("selected_count", 0)
                            )
                            
                            intent = record.get("intent", "unknown")
                            if intent not in stats[log_type]["traces_by_intent"]:
                                stats[log_type]["traces_by_intent"][intent] = 0
                            stats[log_type]["traces_by_intent"][intent] += 1
                        except json.JSONDecodeError:
                            pass
            except Exception:
                pass
    
    return stats
