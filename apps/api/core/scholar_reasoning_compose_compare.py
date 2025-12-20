"""Deterministic compare composer (evidence-only).

Reason: compare questions need concept-complete coverage (definition / practice / confusion)
and should not drift to a single concept.
"""

from __future__ import annotations

from typing import Any

from apps.api.core.schemas import Citation


def _norm(s: str) -> str:
    return (s or "").strip()


def _packet_mentions_concept(p: dict[str, Any], concept_ar: str) -> bool:
    t = _norm(str(p.get("text_ar") or ""))
    c = _norm(concept_ar)
    if not (t and c):
        return False
    return c in t


def compose_compare_answer(
    *,
    question_ar: str,
    concepts_ar: list[str],
    packets: list[dict[str, Any]],
    prefer_more_claims: bool = False,
) -> tuple[str, list[Citation], list[dict[str, Any]]]:
    """
    Build a concept-complete compare matrix.

    Output sections:
    - مصفوفة المقارنة
    - خلاصة تنفيذية (3 نقاط)

    Evidence-only:
    - Each cited line is copied from a packet text.
    - Missing items are explicitly marked "غير منصوص عليه" (meta).
    """

    concepts = [c.strip() for c in (concepts_ar or []) if (c or "").strip()]
    concepts = concepts[:4]

    defs = [p for p in packets if p.get("chunk_type") == "definition"]
    evs = [p for p in packets if p.get("chunk_type") == "evidence"]
    comms = [p for p in packets if p.get("chunk_type") == "commentary"]

    citations: list[Citation] = []
    used_edges: list[dict[str, Any]] = []  # compare has no graph requirement

    def _cite_packet(p: dict[str, Any]) -> None:
        cid = str(p.get("chunk_id") or "")
        if not cid:
            return
        citations.append(
            Citation(
                chunk_id=cid,
                source_anchor=str(p.get("source_anchor") or ""),
                ref=(p.get("refs", [{}])[0].get("ref") if p.get("refs") else None),
            )
        )

    parts: list[str] = []
    parts.append("مصفوفة المقارنة")

    for concept in concepts:
        parts.append(f"- {concept}:")

        # Definition: MUST be grounded if present in packets.
        d = next((p for p in defs if _packet_mentions_concept(p, concept)), None) or (defs[0] if defs else None)
        if d and _norm(str(d.get("text_ar") or "")):
            parts.append(f"- التعريف: {str(d.get('text_ar') or '').strip()}")
            _cite_packet(d)
        else:
            parts.append("- التعريف: غير منصوص عليه")

        # Practical manifestation: prefer commentary / action-like lines.
        c = next((p for p in comms if _packet_mentions_concept(p, concept)), None) or (comms[0] if comms else None)
        if c and _norm(str(c.get("text_ar") or "")):
            parts.append(f"- المظهر العملي: {str(c.get('text_ar') or '').strip()}")
            _cite_packet(c)
        else:
            parts.append("- المظهر العملي: غير منصوص عليه")

        # Common confusion: prefer explicit misunderstanding markers when present.
        mis = next((p for p in comms if "سوء فهم شائع:" in str(p.get("text_ar") or "")), None)
        if mis and _norm(str(mis.get("text_ar") or "")):
            parts.append(f"- الخطأ الشائع: {str(mis.get('text_ar') or '').strip()}")
            _cite_packet(mis)
        else:
            parts.append("- الخطأ الشائع: غير منصوص عليه")

    # BRILLIANCE ENHANCEMENT: Add evidence grounding section
    parts.append("")
    parts.append("التأصيل والأدلة (مختصر ومركز)")
    ev_added = 0
    seen_ev: set[str] = set()
    for p in evs[: (6 if prefer_more_claims else 4)]:
        t = _norm(str(p.get("text_ar") or ""))
        if not t:
            continue
        k = t[:80]
        if k in seen_ev:
            continue
        seen_ev.add(k)
        parts.append(f"- {t}")
        _cite_packet(p)
        ev_added += 1
    if ev_added == 0:
        parts.append("- (راجع التعريفات أعلاه)")
    
    parts.append("")
    parts.append("خلاصة تنفيذية (3 نقاط)")
    summary_src = (defs[:2] + evs[:3] + comms[:3])[: (8 if prefer_more_claims else 6)]
    added = 0
    seen: set[str] = set()
    for p in summary_src:
        if added >= 3:
            break
        t = _norm(str(p.get("text_ar") or ""))
        if not t:
            continue
        k = t[:60]
        if k in seen:
            continue
        seen.add(k)
        parts.append(f"- {t}")
        _cite_packet(p)
        added += 1
    while added < 3:
        parts.append("- غير منصوص عليه")
        added += 1

    return "\n".join(parts).strip(), citations, used_edges

