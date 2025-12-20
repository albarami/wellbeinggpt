"""Breakthrough upgrades for Muḥāsibī reasoning.

Provides:
- Candidate generation + ranking (deterministic, grounded)
- Bounded critic loop (fixable-only triggers)
"""

from apps.api.core.breakthrough.candidate_ranker import (
    Candidate,
    score_candidate,
    pick_best,
)
from apps.api.core.breakthrough.candidate_generator import generate_candidates
from apps.api.core.breakthrough.critic_loop import (
    FIXABLE_FAILURES,
    NON_FIXABLE_FAILURES,
    should_trigger_critic,
    critic_loop_once,
)

__all__ = [
    "Candidate",
    "score_candidate",
    "pick_best",
    "generate_candidates",
    "FIXABLE_FAILURES",
    "NON_FIXABLE_FAILURES",
    "should_trigger_critic",
    "critic_loop_once",
]
