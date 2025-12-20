"""Deterministic partial answer for missing grounded graph links (A+B).

Reason:
- Stakeholder cross-pillar/network/tension questions may be in-scope, but the KB/graph
  may not yet contain grounded semantic edges (edge_justification_span).
- We must remain fail-closed on graph claims while still being useful.

Output:
- Section (A): grounded excerpts (definitions/evidence/commentary) with citations
- Section (B): explicit statement about missing grounded links + what would be required
"""

from __future__ import annotations

from typing import Any

from apps.api.core.schemas import Citation
from apps.api.retrieve.normalize_ar import extract_arabic_words, normalize_for_matching
from apps.api.core.scholar_reasoning_edge_fallback import detect_pillar_ids_from_question


def compose_partial_graph_gap_answer(
    *,
    packets: list[dict[str, Any]],
    question_ar: str,
) -> tuple[str, list[Citation], list[dict[str, Any]]]:
    citations: list[Citation] = []
    used_edges: list[dict[str, Any]] = []

    defs = [p for p in packets if p.get("chunk_type") == "definition"]
    evs = [p for p in packets if p.get("chunk_type") == "evidence"]
    comms = [p for p in packets if p.get("chunk_type") == "commentary"]

    seen_chunk_ids: set[str] = set()

    def _dedupe_packets_by_text(pkts: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Dedupe packets by normalized Arabic text (deterministic)."""

        seen: set[str] = set()
        out: list[dict[str, Any]] = []
        for p in pkts:
            t = " ".join(str(p.get("text_ar") or "").split()).strip()
            if not t:
                continue
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
        return " ".join(cleaned[:max_words]).strip()

    def _score_packet(q_norm: str, p: dict[str, Any], pillar_ids: list[str]) -> float:
        t = str(p.get("text_ar") or "").strip()
        if not t:
            return -1.0
        tn = normalize_for_matching(t)
        if not tn:
            return -1.0
        q_terms = [w for w in (q_norm or "").split() if len(w) >= 3][:24]
        overlap = sum(1 for w in q_terms if w in tn)

        pe = str(p.get("entity_type") or "")
        pid = str(p.get("entity_id") or "")
        pillar_bonus = 0.0
        if pe == "pillar" and pid and pid in set(pillar_ids or []):
            pillar_bonus = 6.0
        boundary_markers = ["حدود", "ضوابط", "ميزان", "انحراف", "افراط", "تفريط"]
        wants_boundary = any(m in (q_norm or "") for m in boundary_markers)
        has_boundary = any(m in tn for m in boundary_markers)
        boundary_bonus = 2.5 if (wants_boundary and has_boundary) else 0.0

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
            pillar_bonus -= 4.0

        return float(overlap) + pillar_bonus + boundary_bonus

    def _cite_packet(p: dict[str, Any]) -> None:
        cid = str(p.get("chunk_id") or "")
        if not cid or cid in seen_chunk_ids:
            return
        seen_chunk_ids.add(cid)
        citations.append(
            Citation(
                chunk_id=cid,
                source_anchor=str(p.get("source_anchor") or ""),
                ref=(p.get("refs", [{}])[0].get("ref") if p.get("refs") else None),
            )
        )

    parts: list[str] = []
    parts.append("ما يمكن دعمه من الأدلة المسترجعة")

    # Grounded excerpts: prioritize definitions, then evidence, then commentaries.
    qn = normalize_for_matching(question_ar or "")
    pillar_ids = detect_pillar_ids_from_question(question_norm=qn)
    candidates = _dedupe_packets_by_text(list(defs or []) + list(evs or []) + list(comms or []))
    candidates.sort(key=lambda p: (-_score_packet(qn, p, pillar_ids), str(p.get("chunk_id") or "")))
    chosen = candidates[:10]

    if not chosen:
        parts.append("- لا يوجد نص مناسب ضمن الأدلة المسترجعة يدعم بناء مسار/شبكة ربط عبر الركائز لهذا السؤال.")
    else:
        for p in chosen:
            t = " ".join(str(p.get("text_ar") or "").split()).strip()
            if not t:
                continue
            parts.append(f"- {t}")
            ex = _extract_excerpt(t, max_words=18)
            if ex:
                parts.append(f"- وجه الدلالة: «{ex}…»")
            _cite_packet(p)

    parts.append("")
    parts.append("ما لا يمكن الجزم به من الأدلة الحالية")
    parts.extend(
        [
            "- لا توجد روابط مؤصّلة (edge_justification_span) في البيانات الحالية يمكن الاعتماد عليها لبناء ربطٍ عبر الركائز وفق متطلبات السؤال (مسار/شبكة/تعارض-توفيق).",
            "- لذلك لا يمكنني ادعاء علاقة (تمكين/تعزيز/شرط/تعارض ظاهري/توفيق) بين قيم من ركائز مختلفة دون نص صريح يربطها ويؤصّلها.",
            "- لا يمكنني إعطاء إجابة كاملة إلا إذا تم إدخال نص/مصدر (أو ملاحظة عالمية) يحتوي على تبرير الربط، ثم حفظه كحافة دلالية مع شاهد/شواهد (edge_justification_span).",
        ]
    )

    return "\n".join(parts).strip(), citations, used_edges

