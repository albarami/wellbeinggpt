"""Rule-based depth/quality rubric (scholar format).

Score /10 (each 0-2):
- Definitions grounded (2)
- Evidence quality (2)
- Cross-pillar justification (2)
- Applied scenario reasoning (2)
- Misunderstandings + executive summary (2)

Notes:
- Deterministic, string/structure based (no LLM).
- This is aligned with the required deep-mode answer format.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from eval.types import EvalOutputRow


@dataclass(frozen=True)
class RubricMetrics:
    average_score: float


def _has_definition_markers(text: str) -> bool:
    t = text or ""
    markers = ["تعريف", "هو", "هي", "يعني", "المقصود"]
    return any(m in t for m in markers)


def _has_section(text: str, header: str) -> bool:
    t = text or ""
    return header in t


def _looks_structured(text: str) -> bool:
    t = text or ""
    return any(x in t for x in ["\n-", "\n1", "خطوات", "أولاً", "ثانياً", "\n\n"]) and len(t) > 120


def _count_bullets_in_section(text: str, header: str) -> int:
    """
    Count '-' bullet lines after a header until the next blank-line-separated header-ish line.

    This is intentionally conservative and deterministic.
    """
    t = text or ""
    if header not in t:
        return 0
    lines = [ln.rstrip() for ln in t.splitlines()]
    start = None
    for i, ln in enumerate(lines):
        if header in ln:
            start = i + 1
            break
    if start is None:
        return 0
    bullets = 0
    for ln in lines[start:]:
        s = ln.strip()
        if not s:
            # Stop when we hit an empty line and the next non-empty is likely another section header.
            continue
        # Heuristic: section headers are non-bulleted lines with no leading dash and short-ish.
        if (not s.startswith("-")) and (len(s) <= 40) and (" " in s or "داخل" in s or "خلاصة" in s):
            break
        if s.startswith("-"):
            bullets += 1
    return bullets


def score_rubric(outputs: list[EvalOutputRow], dataset_by_id: dict[str, dict[str, Any]]) -> RubricMetrics:
    scores: list[int] = []

    for r in outputs:
        d = dataset_by_id.get(r.id, {})
        scores.append(score_rubric_row(r, d))

    return RubricMetrics(average_score=(sum(scores) / len(scores)) if scores else 0.0)


def score_rubric_row(r: EvalOutputRow, d: dict[str, Any]) -> int:
    """Score a single row (0-10)."""
    if r.abstained:
        return 0

    s = 0

    ans = r.answer_ar or ""

    # 1) Definitions grounded (2)
    if _has_section(ans, "تعريف المفهوم داخل الإطار") and (len(r.citations) >= 1 or _has_definition_markers(ans)):
        s += 2

    # 2) Evidence quality (2)
    # Require evidence section + at least 2 citations.
    if _has_section(ans, "التأصيل والأدلة") and len(r.citations) >= 2:
        s += 2

    # 3) Cross-pillar justification (2)
    if str(d.get("type")) == "cross_pillar":
        if _has_section(ans, "الربط بين الركائز") and (r.graph_trace.edges or r.graph_trace.paths) and len(r.citations) >= 2:
            s += 2
    else:
        # Non-cross questions: do not penalize.
        s += 2 if _looks_structured(ans) else 0

    # 4) Applied scenario reasoning (2)
    if _has_section(ans, "تطبيق عملي") and _count_bullets_in_section(ans, "تطبيق عملي") >= 2 and len(r.citations) >= 2:
        s += 2

    # 5) Misunderstandings + executive summary (2)
    has_warn = _has_section(ans, "تنبيهات وأخطاء شائعة") and _count_bullets_in_section(ans, "تنبيهات وأخطاء شائعة") >= 1
    has_exec = _has_section(ans, "خلاصة تنفيذية") and _count_bullets_in_section(ans, "خلاصة تنفيذية") >= 3
    if has_warn and has_exec and len(r.citations) >= 2:
        s += 2

    return min(10, s)
