"""
Muḥāsibī trace utilities (safe, non chain-of-thought).

We expose high-level state transitions, timings, and counts.
We do NOT expose hidden reasoning or internal chain-of-thought.
"""

from __future__ import annotations

from typing import Any


def summarize_state(state_name: str, ctx) -> dict[str, Any]:
    """
    Build a safe trace snapshot for a given state.
    """
    base: dict[str, Any] = {
        "state": state_name,
        "mode": getattr(ctx, "mode", "answer"),
        "language": getattr(ctx, "language", "ar"),
    }

    if state_name == "LISTEN":
        base.update(
            {
                "detected_entities_count": len(getattr(ctx, "detected_entities", []) or []),
                "keywords_count": len(getattr(ctx, "question_keywords", []) or []),
                "listen_summary_ar": getattr(ctx, "listen_summary_ar", ""),
            }
        )
    elif state_name == "PURPOSE":
        p = getattr(ctx, "purpose", None)
        base.update(
            {
                "ultimate_goal_ar": getattr(p, "ultimate_goal_ar", None),
                "constraints_count": len(getattr(p, "constraints_ar", []) or []) if p else 0,
            }
        )
    elif state_name == "PATH":
        base.update({"path_steps": list(getattr(ctx, "path_plan_ar", []) or [])[:6]})
    elif state_name == "RETRIEVE":
        base.update(
            {
                "evidence_packets_count": len(getattr(ctx, "evidence_packets", []) or []),
                "has_definition": bool(getattr(ctx, "has_definition", False)),
                "has_evidence": bool(getattr(ctx, "has_evidence", False)),
            }
        )
    elif state_name == "ACCOUNT":
        base.update(
            {
                "not_found": bool(getattr(ctx, "not_found", False)),
                "issues": list(getattr(ctx, "account_issues", []) or [])[:6],
            }
        )
    elif state_name == "INTERPRET":
        base.update(
            {
                "not_found": bool(getattr(ctx, "not_found", False)),
                "confidence": str(getattr(ctx, "confidence", "")),
                "citations_count": len(getattr(ctx, "citations", []) or []),
            }
        )
    elif state_name == "REFLECT":
        base.update({"reflection_added": bool(getattr(ctx, "reflection_added", False))})
    elif state_name == "FINALIZE":
        base.update({"done": True})

    return base


