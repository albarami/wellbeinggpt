"""
Edge Candidate Logger - Logs candidate edge pools for training data collection.

Purpose:
- Log per-query candidate edge sets (before selection)
- Track selected vs rejected edges
- Collect real training labels for edge scorer

Only logs for PASS_FULL outcomes on relevant intents.
"""

import json
import logging
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Configuration
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
TRACE_DIR = PROJECT_ROOT / "data" / "phase2" / "edge_traces"

# Intents that benefit from edge scoring
EDGE_SCORING_INTENTS = {
    "cross_pillar",
    "cross_pillar_path",
    "network_build",
    "global_synthesis",
    "world_model_synthesis",
    "pillar_relationship",
}


def should_log_edges(intent: Optional[str]) -> bool:
    """Determine if candidate edges should be logged for this intent."""
    if not os.getenv("EDGE_TRACE_LOGGING", "false").lower() in {"1", "true", "yes"}:
        return False
    
    intent_lower = (intent or "").lower().strip()
    return intent_lower in EDGE_SCORING_INTENTS


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
    
    Only logs PASS_FULL outcomes to ensure clean labels.
    
    Args:
        request_id: Unique request identifier
        question: The user question
        intent: Detected intent type
        mode: Request mode (answer, natural_chat, etc.)
        candidate_edges: All candidate edges (before selection)
        selected_edge_ids: IDs of edges that were selected (used_edges)
        contract_outcome: PASS_FULL, PASS_PARTIAL, or FAIL
    """
    # Only log PASS_FULL for clean training labels
    if contract_outcome != "PASS_FULL":
        return
    
    if not should_log_edges(intent):
        return
    
    try:
        TRACE_DIR.mkdir(parents=True, exist_ok=True)
        
        # Compute quality scores for all candidates
        enriched_candidates = []
        for edge in candidate_edges:
            edge_copy = edge.copy()
            edge_copy["quality_score"] = compute_edge_quality_score(edge)
            enriched_candidates.append(edge_copy)
        
        selected_set = set(selected_edge_ids)
        
        record = {
            "trace_id": str(uuid.uuid4()),
            "timestamp": datetime.utcnow().isoformat(),
            "request_id": request_id,
            "question": question,
            "intent": intent,
            "mode": mode,
            "contract_outcome": contract_outcome,
            "candidate_count": len(candidate_edges),
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
                    "is_selected": str(e.get("edge_id", "")) in selected_set,
                }
                for e in enriched_candidates
            ],
            "selected_edge_ids": selected_edge_ids,
            "rejected_edge_ids": [
                str(e.get("edge_id", ""))
                for e in candidate_edges
                if str(e.get("edge_id", "")) not in selected_set
            ],
        }
        
        # Write to daily file
        today = datetime.utcnow().strftime("%Y%m%d")
        trace_file = TRACE_DIR / f"edge_traces_{today}.jsonl"
        
        with open(trace_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        
        logger.debug(f"Logged {len(candidate_edges)} candidate edges for {intent}")
        
    except Exception as e:
        logger.warning(f"Failed to log candidate edges: {e}")


def get_edge_trace_stats() -> dict[str, Any]:
    """Get statistics about collected edge traces."""
    stats = {
        "total_traces": 0,
        "total_candidates": 0,
        "total_selected": 0,
        "total_rejected": 0,
        "traces_by_intent": {},
        "trace_files": [],
    }
    
    if not TRACE_DIR.exists():
        return stats
    
    for trace_file in TRACE_DIR.glob("edge_traces_*.jsonl"):
        stats["trace_files"].append(str(trace_file.name))
        with open(trace_file, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    record = json.loads(line)
                    stats["total_traces"] += 1
                    stats["total_candidates"] += record.get("candidate_count", 0)
                    stats["total_selected"] += record.get("selected_count", 0)
                    stats["total_rejected"] += record.get("candidate_count", 0) - record.get("selected_count", 0)
                    
                    intent = record.get("intent", "unknown")
                    if intent not in stats["traces_by_intent"]:
                        stats["traces_by_intent"][intent] = 0
                    stats["traces_by_intent"][intent] += 1
                except:
                    pass
    
    return stats
