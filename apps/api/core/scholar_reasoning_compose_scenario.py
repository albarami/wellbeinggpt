"""Deterministic partial scenario composer (evidence-only).

Stakeholder-friendly behavior:
- Provide what can be supported (A)
- Explicitly list what cannot be supported (B)
- Do not ask follow-up questions unless explicitly requested
"""

from __future__ import annotations

from typing import Any

from apps.api.core.schemas import Citation
from apps.api.retrieve.normalize_ar import extract_arabic_words, normalize_for_matching
from apps.api.core.scholar_reasoning_edge_fallback import detect_pillar_ids_from_question


def compose_partial_scenario_answer(
    *,
    packets: list[dict[str, Any]],
    question_ar: str = "",
    prefer_more_claims: bool = False,
) -> tuple[str, list[Citation], list[dict[str, Any]]]:
    citations: list[Citation] = []
    used_edges: list[dict[str, Any]] = []

    defs = [p for p in packets if p.get("chunk_type") == "definition"]
    evs = [p for p in packets if p.get("chunk_type") == "evidence"]
    comms = [p for p in packets if p.get("chunk_type") == "commentary"]

    def _dedupe_packets_by_text(pkts: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """
        Dedupe packets by normalized Arabic text (deterministic).

        Reason:
        - The DB can legitimately contain repeated hadith/ayah text as separate chunks.
        - Showing the same line multiple times is perceived as "blind" and adds no depth.
        """

        seen: set[str] = set()
        out: list[dict[str, Any]] = []
        for p in pkts:
            t = " ".join(str(p.get("text_ar") or "").split()).strip()
            if not t:
                continue
            # Skip tiny fragments (often titles like "الاهتمام" that feel random).
            if len(extract_arabic_words(t)) < 3 and len(t) < 40:
                continue
            key = normalize_for_matching(t)
            if not key or key in seen:
                continue
            seen.add(key)
            out.append(p)
        return out

    def _extract_excerpt(text_ar: str, *, max_words: int = 18) -> str:
        t = (text_ar or "").strip().replace("\n", " ")
        if not t:
            return ""
        cleaned: list[str] = []
        for w in t.split():
            ww = w.strip("()[]{}\"'«»،,.;:!?…ـ")
            if len(ww) >= 3:
                cleaned.append(ww)
        words = cleaned
        return " ".join(words[:max_words]).strip()

    def _score_packet(q_norm: str, p: dict[str, Any], pillar_ids: list[str]) -> float:
        """
        Deterministic relevance scoring for fallback selection.

        Signals:
        - term overlap between question and packet text
        - pillar alignment (pillar:P00x packets that match question pillars)
        - boundary-marker alignment (حدود/ضوابط/انحراف...)
        """

        t = str(p.get("text_ar") or "").strip()
        if not t:
            return -1.0
        tn = normalize_for_matching(t)
        if not tn:
            return -1.0
        q_terms = [w for w in (q_norm or "").split() if len(w) >= 3][:24]
        overlap = sum(1 for w in q_terms if w in tn)

        # Pillar alignment: if the question mentions a pillar and packet is that pillar, boost.
        pe = str(p.get("entity_type") or "")
        pid = str(p.get("entity_id") or "")
        pillar_bonus = 0.0
        if pe == "pillar" and pid and pid in set(pillar_ids or []):
            pillar_bonus = 6.0
        elif any(k in tn for k in ["الحياة البدنية", "الصحة", "الجسد", "الروح", "العاطف", "الفكر", "الاجتماع"]):
            # Soft bonus for pillar language in the packet when the question contains it.
            pillar_bonus = 1.5 if any(k in (q_norm or "") for k in ["الحياة البدنية", "الصحة", "الجسد", "الروح", "العاطف", "الفكر", "الاجتماع"]) else 0.0

        boundary_markers = ["حدود", "ضوابط", "ميزان", "انحراف", "افراط", "تفريط"]
        wants_boundary = any(m in (q_norm or "") for m in boundary_markers)
        has_boundary = any(m in tn for m in boundary_markers)
        boundary_bonus = 2.5 if (wants_boundary and has_boundary) else 0.0

        # Prefer longer, contentful snippets slightly (but don't overfit length).
        length_bonus = min(1.5, max(0.0, len(tn) / 220.0))

        # Penalize "wrong-pillar" packets when question mentions a pillar but packet has none of its keywords.
        pillar_kw = {
            "P001": ["الحياة الروحية", "روحي", "الروح", "ايمان", "الايمان", "قلب"],
            "P002": ["الحياة العاطفية", "عاطف", "مشاعر", "نفس"],
            "P003": ["الحياة الفكرية", "فكر", "عقل", "معرف"],
            "P004": ["الحياة البدنية", "بدني", "الصحة", "الجسد", "الجسم", "عافية"],
            "P005": ["الحياة الاجتماعية", "اجتماع", "مجتمع", "اسري", "أسري", "جماعي"],
        }
        need_kw: list[str] = []
        for pid2 in (pillar_ids or [])[:2]:
            need_kw.extend(pillar_kw.get(pid2, []))
        if need_kw and not any(k in tn for k in need_kw):
            # If question is pillar-specific and packet doesn't reflect it, demote (still allow if nothing else exists).
            pillar_bonus -= 4.0

        return float(overlap) + pillar_bonus + boundary_bonus + length_bonus

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
    parts.append("ما يمكن دعمه من الأدلة المسترجعة")

    # Grounded highlights: definitions + evidence + actionable guidance (verbatim packets).
    qn = normalize_for_matching(question_ar or "")
    pillar_ids = detect_pillar_ids_from_question(question_norm=qn)
    candidates = _dedupe_packets_by_text(list(defs or []) + list(evs or []) + list(comms or []))
    candidates.sort(key=lambda p: (-_score_packet(qn, p, pillar_ids), str(p.get("chunk_id") or "")))
    max_items = 10 if prefer_more_claims else 7
    chosen = candidates[:max_items]

    if not chosen:
        parts.append("- لا يوجد نص مناسب ضمن الأدلة المسترجعة لدعم تحليل الحالة.")
    else:
        for p in chosen:
            t = " ".join(str(p.get("text_ar") or "").split()).strip()
            if not t:
                continue
            parts.append(f"- {t}")
            # Evidence-to-claim synthesis (deterministic, bindable): reuse words from the same quote.
            ex = _extract_excerpt(t, max_words=18)
            if ex:
                # Put excerpt early so binding uses quote terms (claim binder caps at 20 terms).
                parts.append(f"- وجه الدلالة: «{ex}…»")
            _cite_packet(p)

    parts.append("")
    parts.append("ما لا يمكن الجزم به من الأدلة الحالية")
    # Meta-only limits (no claims beyond "not supported by current evidence set").
    parts.extend(
        [
            "- لا يمكنني الجزم بعلاقة سببية بين الأعراض المذكورة وبين قيمة بعينها بدون نص صريح ضمن الأدلة.",
            "- لا يمكنني تقديم تشخيص طبي أو تفسير مرضي للأعراض من هذا النظام.",
            "- لا يمكنني تقديم مؤشرات قياس تفصيلية لكل خطوة إذا لم يرد نص صريح يحددها ضمن الأدلة المسترجعة.",
        ]
    )

    return "\n".join(parts).strip(), citations, used_edges

