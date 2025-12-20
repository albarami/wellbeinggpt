"""Validation helpers for eval harness.

These are *hard gates* used by tests and scorers.
"""

from __future__ import annotations

from typing import Iterable

from eval.types import EvalOutputRow


def validate_eval_rows(rows: Iterable[EvalOutputRow]) -> list[str]:
    """Validate basic invariants and return list of issues."""
    issues: list[str] = []
    for r in rows:
        # Basic field sanity
        if not r.id:
            issues.append("row.id is empty")
        if not r.question:
            issues.append(f"{r.id}: question empty")

        # Abstention invariants
        if r.abstained and not (r.abstain_reason or "").strip():
            issues.append(f"{r.id}: abstained=true requires abstain_reason")

        # If not abstained, answer must be non-empty.
        if (not r.abstained) and not (r.answer_ar or "").strip():
            issues.append(f"{r.id}: abstained=false but answer_ar empty")

        # Claims must have ids.
        for c in r.claims:
            if not c.claim_id:
                issues.append(f"{r.id}: claim missing claim_id")

    return issues
