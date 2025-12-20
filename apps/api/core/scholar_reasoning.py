"""Deterministic scholar reasoning (deep + light deep).

This file is intentionally kept <500 LOC.
Implementation lives in:
- `scholar_reasoning_impl.py` (retrieval expansion + gates)
- `scholar_reasoning_compose.py` (deterministic answer composition)
"""

from __future__ import annotations

from apps.api.core.scholar_reasoning_impl import ScholarDepthTargets, ScholarReasoner

__all__ = ["ScholarDepthTargets", "ScholarReasoner"]
