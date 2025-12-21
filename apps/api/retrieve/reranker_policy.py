"""
Reranker Policy: Determines when to enable the reranker.

Based on A/B test results:
- Global reranker ON hurts overall PASS_FULL (-3.7%)
- Reranker helps synthesis questions (synth-006: 30% → 100%)

Policy:
- Enable reranker selectively for synthesis/complex intents
- Enable conditionally when first-pass retrieval looks weak
- Disable for simple questions where it causes regression
"""

from typing import Any, Optional
import logging

logger = logging.getLogger(__name__)

# Intents that ALWAYS benefit from reranker
ALWAYS_RERANK_INTENTS = frozenset([
    "global_synthesis",
    "world_model_synthesis",
    "network_build",
    "GLOBAL_SYNTHESIS_WORLD_MODEL",
])

# Intents that MAY benefit from reranker (conditional)
CONDITIONAL_RERANK_INTENTS = frozenset([
    "cross_pillar",
    "cross_pillar_path",
    "pillar_relationship",
])

# Intents that should NOT use reranker (causes regression)
NO_RERANK_INTENTS = frozenset([
    "natural_chat",
    "guidance_framework_chat",
    "boundaries",
    "system_limits_policy",
    "compare_definition",
    "definition",
    "stakeholder_specific",
])

# Thresholds for conditional reranking
TOP1_SCORE_THRESHOLD = 0.6  # Rerank if top score below this
MEAN_TOPK_THRESHOLD = 0.4   # Rerank if mean top-k below this
MIN_DISTINCT_SOURCES = 2    # Rerank if fewer than this many sources


def should_use_reranker(
    intent: Optional[str] = None,
    retrieval_scores: Optional[list[float]] = None,
    retrieval_sources: Optional[list[str]] = None,
    force_on: bool = False,
    force_off: bool = False,
    mode: Optional[str] = None,
) -> tuple[bool, str]:
    """
    Determine whether to use the reranker for this request.
    
    Args:
        intent: The detected intent type (e.g., "global_synthesis", "natural_chat")
        retrieval_scores: List of scores from first-pass retrieval (highest first)
        retrieval_sources: List of source identifiers from retrieval
        force_on: Override to always enable
        force_off: Override to always disable
        mode: The question mode (natural_chat, answer, etc.)
    
    Returns:
        tuple[bool, str]: (should_use, reason)
    
    Reason: Per-intent gating based on A/B test results showing reranker helps
    synthesis but hurts overall breadth.
    """
    # Handle explicit overrides
    if force_off:
        return False, "force_off"
    if force_on:
        return True, "force_on"
    
    # HARD RULE: natural_chat mode NEVER uses reranker
    # Reason: A/B test showed reranker causes regression on natural_chat questions
    mode_lower = (mode or "").lower().strip()
    if mode_lower == "natural_chat":
        logger.debug(f"Reranker OFF: mode=natural_chat (hard rule)")
        return False, "mode_natural_chat"
    
    # Normalize intent
    intent_lower = (intent or "").lower().strip()
    
    # HARD RULE: guidance_framework_chat NEVER uses reranker
    if intent_lower == "guidance_framework_chat":
        logger.debug(f"Reranker OFF: intent=guidance_framework_chat (hard rule)")
        return False, "intent_guidance_chat"
    
    # Check ALWAYS_RERANK intents
    if intent_lower in ALWAYS_RERANK_INTENTS or intent in ALWAYS_RERANK_INTENTS:
        logger.debug(f"Reranker ON: intent '{intent}' in ALWAYS_RERANK")
        return True, f"always_rerank_intent_{intent_lower}"
    
    # Check NO_RERANK intents
    if intent_lower in NO_RERANK_INTENTS or intent in NO_RERANK_INTENTS:
        logger.debug(f"Reranker OFF: intent '{intent}' in NO_RERANK")
        return False, f"no_rerank_intent_{intent_lower}"
    
    # Check CONDITIONAL intents with retrieval quality triggers
    if intent_lower in CONDITIONAL_RERANK_INTENTS or intent in CONDITIONAL_RERANK_INTENTS:
        should_rerank = _check_retrieval_quality_triggers(
            retrieval_scores, retrieval_sources
        )
        reason = "conditional_triggered" if should_rerank else "conditional_not_triggered"
        return should_rerank, reason
    
    # Default: don't rerank (conservative, matches OFF baseline)
    logger.debug(f"Reranker OFF: intent '{intent}' not in known lists, using default OFF")
    return False, "default_off"


def _check_retrieval_quality_triggers(
    scores: Optional[list[float]],
    sources: Optional[list[str]],
) -> bool:
    """
    Check if retrieval quality triggers warrant reranking.
    
    Returns True if retrieval looks weak and reranking might help.
    """
    if not scores or len(scores) == 0:
        # No retrieval scores - rerank might help
        logger.debug("Reranker ON: no retrieval scores available")
        return True
    
    top1_score = scores[0] if scores else 0.0
    mean_topk = sum(scores[:5]) / min(5, len(scores)) if scores else 0.0
    
    # Trigger 1: Top score is low
    if top1_score < TOP1_SCORE_THRESHOLD:
        logger.debug(f"Reranker ON: top1_score {top1_score:.2f} < {TOP1_SCORE_THRESHOLD}")
        return True
    
    # Trigger 2: Mean top-k is low
    if mean_topk < MEAN_TOPK_THRESHOLD:
        logger.debug(f"Reranker ON: mean_topk {mean_topk:.2f} < {MEAN_TOPK_THRESHOLD}")
        return True
    
    # Trigger 3: Low source diversity
    if sources:
        distinct_sources = len(set(sources[:5]))
        if distinct_sources < MIN_DISTINCT_SOURCES:
            logger.debug(f"Reranker ON: distinct_sources {distinct_sources} < {MIN_DISTINCT_SOURCES}")
            return True
    
    # All triggers passed - retrieval looks good, don't rerank
    logger.debug("Reranker OFF: retrieval quality looks good")
    return False


def get_reranker_decision_reason(
    intent: Optional[str] = None,
    retrieval_scores: Optional[list[float]] = None,
    retrieval_sources: Optional[list[str]] = None,
) -> str:
    """
    Get a human-readable explanation of the reranker decision.
    For observability and debugging.
    """
    intent_lower = (intent or "").lower().strip()
    
    if intent_lower in ALWAYS_RERANK_INTENTS or intent in ALWAYS_RERANK_INTENTS:
        return f"ALWAYS_ON: intent={intent}"
    
    if intent_lower in NO_RERANK_INTENTS or intent in NO_RERANK_INTENTS:
        return f"ALWAYS_OFF: intent={intent}"
    
    if intent_lower in CONDITIONAL_RERANK_INTENTS or intent in CONDITIONAL_RERANK_INTENTS:
        if not retrieval_scores:
            return "CONDITIONAL_ON: no scores"
        
        top1 = retrieval_scores[0] if retrieval_scores else 0.0
        mean_k = sum(retrieval_scores[:5]) / min(5, len(retrieval_scores)) if retrieval_scores else 0.0
        distinct = len(set((retrieval_sources or [])[:5]))
        
        if top1 < TOP1_SCORE_THRESHOLD:
            return f"CONDITIONAL_ON: top1={top1:.2f} < {TOP1_SCORE_THRESHOLD}"
        if mean_k < MEAN_TOPK_THRESHOLD:
            return f"CONDITIONAL_ON: mean_topk={mean_k:.2f} < {MEAN_TOPK_THRESHOLD}"
        if distinct < MIN_DISTINCT_SOURCES:
            return f"CONDITIONAL_ON: distinct_sources={distinct} < {MIN_DISTINCT_SOURCES}"
        
        return f"CONDITIONAL_OFF: retrieval quality OK (top1={top1:.2f}, mean={mean_k:.2f})"
    
    return f"DEFAULT_OFF: unknown intent={intent}"
