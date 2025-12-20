"""Scholar answer composition helpers (evidence-only).

Reason: keep each module <500 LOC (see PLANNING.md).
"""

from __future__ import annotations

from typing import Any

from apps.api.core.schemas import Citation
from apps.api.retrieve.normalize_ar import normalize_for_matching


def _dedupe_packets_by_text(pkts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Dedupe packets by normalized Arabic text (deterministic).

    Reason:
    - The framework ingestion may store repeated ayah/hadith text across multiple chunks.
    - Repeating identical lines makes answers feel "blind" and reduces perceived intelligence.
    """

    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for p in pkts:
        t = str(p.get("text_ar") or "").strip()
        if not t:
            continue
        key = normalize_for_matching(t)
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(p)
    return out


def _is_relevant_for_global_synthesis(p: dict[str, Any]) -> bool:
    """
    Filter evidence for global_synthesis: prefer pillar defs, cross-pillar integration,
    boundaries, and purpose statements. Penalize generic anthropology verses.
    
    Reason: Increases perceived intelligence by filtering irrelevant evidence.
    """
    t = str(p.get("text_ar") or "").strip().lower()
    if not t:
        return False
    
    # High-relevance markers (keep these)
    high_relevance = [
        "ركيزة", "قيمة", "تكامل", "توازن", "حياة طيبة", "ازدهار",
        "غاية", "وظيفة", "هدف", "مقصد", "حد", "حدود", "ضابط",
        "تعريف", "معنى", "مفهوم", "علاقة", "ارتباط", "صلة",
        "بدني", "فكري", "روحي", "عاطفي", "اجتماعي", "مادي",
    ]
    
    # Low-relevance markers (generic anthropology - penalize unless explicitly used)
    low_relevance = [
        "عجول", "من عجل", "خلق الإنسان",  # Generic human nature
    ]
    
    # Check for low-relevance markers first
    for marker in low_relevance:
        if marker in t:
            # Only include if it also has high-relevance context
            for hr in high_relevance:
                if hr in t:
                    return True
            return False
    
    return True  # Default: include


def _paraphrase_text(t: str) -> str:
    """
    Create a scholarly paraphrase from quoted text.
    
    Reason: Quote budget requires paraphrase+cite for non-quoted evidence.
    """
    # Extract key terms and create a scholarly summary
    cleaned: list[str] = []
    for w in t.split():
        ww = w.strip("()[]{}\"'«»،,.;:!?…ـ")
        if len(ww) >= 3:
            cleaned.append(ww)
    
    if len(cleaned) < 5:
        return t  # Too short to paraphrase
    
    # Create a scholarly paraphrase prefix
    key_terms = " ".join(cleaned[:12])
    return f"يُستفاد من النص: {key_terms}..."


def compose_deep_answer(
    *,
    packets: list[dict[str, Any]],
    semantic_edges: list[dict[str, Any]],
    max_edges: int,
    question_ar: str = "",
    prefer_more_claims: bool = False,
) -> tuple[str, list[Citation], list[dict[str, Any]]]:
    """Compose the required scholar-format deep answer using only evidence text."""

    packets = _dedupe_packets_by_text(list(packets or []))
    
    # RELEVANCE FILTER: Apply for global_synthesis-type questions
    # Reason: Increases perceived intelligence by filtering irrelevant evidence
    if any(k in (question_ar or "").lower() for k in ["ازدهار", "حياة طيبة", "إطار", "ركائز"]):
        packets = [p for p in packets if _is_relevant_for_global_synthesis(p)]
    
    defs = [p for p in packets if p.get("chunk_type") == "definition"]
    evs = [p for p in packets if p.get("chunk_type") == "evidence"]
    comms = [p for p in packets if p.get("chunk_type") == "commentary"]

    defs_n = 4 if prefer_more_claims else 3
    evs_n = 10 if prefer_more_claims else 8
    comms_n = 14 if prefer_more_claims else 10

    citations: list[Citation] = []
    used_edges: list[dict[str, Any]] = []
    used_text: set[str] = set()
    
    # QUOTE BUDGET: Max 6 direct quotes in answer mode, paraphrase the rest
    # Reason: Makes answers feel more intelligent, less mechanical
    quote_budget = 6
    quotes_used = 0

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

    def _append_unique_packet(p: dict[str, Any], force_paraphrase: bool = False) -> bool:
        """Append packet text once across the whole answer (deterministic)."""
        nonlocal quotes_used

        t = str(p.get("text_ar") or "").replace("\r", " ").replace("\n", " ").strip()
        if not t:
            return False
        key = normalize_for_matching(t)
        if not key or key in used_text:
            return False
        used_text.add(key)
        
        # QUOTE BUDGET: Use direct quote only if budget allows, else paraphrase
        if not force_paraphrase and quotes_used < quote_budget:
            parts.append(f"- {t}")
            quotes_used += 1
            # Evidence-to-claim synthesis: one bound sentence derived from the same quote text.
            # Reason: "وجه الدلالة" is what makes it read like a scholar while staying bindable.
            cleaned: list[str] = []
            for w in t.split():
                ww = w.strip("()[]{}\"'«»،,.;:!?…ـ")
                if len(ww) >= 3:
                    cleaned.append(ww)
            ex = " ".join(cleaned[:18]).strip()
            if ex:
                parts.append(f"- وجه الدلالة: «{ex}…»")
        else:
            # Paraphrase mode: scholarly summary + cite
            paraphrased = _paraphrase_text(t)
            parts.append(f"- {paraphrased}")
        
        _cite_packet(p)
        return True

    parts: list[str] = []
    parts.append("تعريف المفهوم داخل الإطار")
    if defs:
        added = 0
        for d in defs[:defs_n]:
            if _append_unique_packet(d):
                added += 1
        if added == 0:
            parts.append("- (لا يوجد نص تعريف صريح ضمن الأدلة المسترجعة)")
    else:
        parts.append("- (لا يوجد نص تعريف صريح ضمن الأدلة المسترجعة)")

    parts.append("")
    parts.append("التأصيل والأدلة (مختصر ومركز)")
    if evs:
        added = 0
        for e in evs[:evs_n]:
            if _append_unique_packet(e):
                added += 1
        if added == 0:
            parts.append("- (لا يوجد نص تأصيلي صريح ضمن الأدلة المسترجعة)")
    else:
        parts.append("- (لا يوجد نص تأصيلي صريح ضمن الأدلة المسترجعة)")

    parts.append("")
    parts.append("الربط بين الركائز (مع سبب الربط)")
    if semantic_edges:
        for ed in semantic_edges[:max_edges]:
            rt = str(ed.get("relation_type") or "").strip()
            n_type = str(ed.get("neighbor_type") or "").strip()
            n_id = str(ed.get("neighbor_id") or "").strip()
            spans = list(ed.get("justification_spans") or [])
            edge_id = str(ed.get("edge_id") or "").strip()
            src_type = str(ed.get("source_type") or "").strip()
            src_id = str(ed.get("source_id") or "").strip()
            direction = str(ed.get("direction") or "").strip()
            if not (rt and spans and n_type and n_id):
                continue
            picked_spans = (spans[:2] if prefer_more_claims else spans[:1])
            recorded_spans: list[dict[str, Any]] = []
            for sp in picked_spans:
                quote = str(sp.get("quote") or "").strip()
                q_chunk_id = str(sp.get("chunk_id") or "").strip()
                if quote:
                    parts.append(f"- ({rt}) رابط إلى {n_type}:{n_id} — شاهد: {quote}")
                    if q_chunk_id:
                        citations.append(Citation(chunk_id=q_chunk_id, source_anchor="", ref=None))
                    recorded_spans.append(
                        {
                            "chunk_id": q_chunk_id,
                            "span_start": int(sp.get("span_start") or 0),
                            "span_end": int(sp.get("span_end") or 0),
                            "quote": quote,
                        }
                    )
            # Record the edge as "used" iff we emitted at least one justification line.
            if recorded_spans and edge_id and src_type and src_id:
                from_node = f"{src_type}:{src_id}"
                to_node = f"{n_type}:{n_id}"
                if direction == "incoming":
                    # Incoming neighbor means edge direction in DB is neighbor -> source.
                    from_node, to_node = to_node, from_node
                used_edges.append(
                    {
                        "edge_id": edge_id,
                        "from_node": from_node,
                        "to_node": to_node,
                        "relation_type": rt,
                        "justification_spans": recorded_spans,
                    }
                )
    else:
        parts.append("- (لا توجد روابط دلالية مُبرَّرة ضمن الأدلة المسترجعة)")

    parts.append("")
    parts.append("تطبيق عملي على الحالة/السؤال")
    scenario_like = [c for c in comms[:comms_n] if "سيناريو:" in str(c.get("text_ar") or "")]
    action_like = [
        c
        for c in comms[:comms_n]
        if any(k in str(c.get("text_ar") or "") for k in ["ينبغي", "احرص", "تدرّب", "تدرب", "طبّق", "طبق", "خطوة", "خطوات"])
    ]
    chosen_app = (scenario_like[:4] or action_like[:4] or comms[:4])[: (4 if prefer_more_claims else 3)]
    if len(chosen_app) < 2:
        chosen_app = list(chosen_app) + evs[: max(0, 2 - len(chosen_app))]
    added = 0
    for c in chosen_app:
        if _append_unique_packet(c):
            added += 1
    if added == 0:
        parts.append("- غير منصوص عليه في الأدلة المسترجعة.")

    parts.append("")
    parts.append("تنبيهات وأخطاء شائعة")
    mis_like = [c for c in comms[:comms_n] if "سوء فهم شائع:" in str(c.get("text_ar") or "")]
    warn_like = [c for c in comms[:comms_n] if any(k in str(c.get("text_ar") or "") for k in ["تنبيه", "خطأ", "احذر", "تحذير"])]
    chosen_warn = (mis_like[:3] or warn_like[:3] or comms[3:8])[: (3 if prefer_more_claims else 2)]
    if len(chosen_warn) < 1:
        chosen_warn = list(chosen_warn) + evs[:1]
    added = 0
    for c in chosen_warn:
        if _append_unique_packet(c):
            added += 1
    if added == 0:
        parts.append("- غير منصوص عليه في الأدلة المسترجعة.")

    parts.append("")
    parts.append("خلاصة تنفيذية (3 نقاط)")
    summary_src = (defs[:2] + evs[:3] + comms[:3])
    added = 0
    for p in summary_src:
        if added >= 3:
            break
        if _append_unique_packet(p):
            added += 1
    while added < 3:
        parts.append("- غير منصوص عليه")
        added += 1

    return "\n".join(parts).strip(), citations, used_edges


def compose_light_answer(
    *,
    packets: list[dict[str, Any]],
    prefer_more_claims: bool = False,
) -> tuple[str, list[Citation]]:
    """Compose a shorter scholar-format answer (for Gold QA) using only evidence text."""

    packets = _dedupe_packets_by_text(list(packets or []))
    defs = [p for p in packets if p.get("chunk_type") == "definition"]
    evs = [p for p in packets if p.get("chunk_type") == "evidence"]
    comms = [p for p in packets if p.get("chunk_type") == "commentary"]

    defs_n = 3 if prefer_more_claims else 2
    evs_n = 6 if prefer_more_claims else 4
    app_n = 3 if prefer_more_claims else 2

    citations: list[Citation] = []
    used_text: set[str] = set()

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

    def _append_unique_packet(p: dict[str, Any]) -> bool:
        t = str(p.get("text_ar") or "").strip()
        if not t:
            return False
        key = normalize_for_matching(t)
        if not key or key in used_text:
            return False
        used_text.add(key)
        parts.append(f"- {t}")
        _cite_packet(p)
        return True

    parts: list[str] = []
    parts.append("تعريف المفهوم داخل الإطار")
    added = 0
    for d in defs[:defs_n]:
        if _append_unique_packet(d):
            added += 1
    if added == 0:
        parts.append("- غير منصوص عليه")

    parts.append("")
    parts.append("التأصيل والأدلة (مختصر ومركز)")
    added = 0
    for e in evs[:evs_n]:
        if _append_unique_packet(e):
            added += 1
    if added == 0:
        parts.append("- غير منصوص عليه")

    parts.append("")
    parts.append("تطبيق عملي على الحالة/السؤال")
    action_like = [
        c
        for c in comms
        if any(k in str(c.get("text_ar") or "") for k in ["ينبغي", "احرص", "تدرّب", "تدرب", "طبّق", "طبق", "خطوة", "خطوات", "سيناريو:"])
    ]
    chosen_app = (action_like[:app_n] or comms[:app_n] or evs[:app_n])
    added = 0
    for c in chosen_app:
        if _append_unique_packet(c):
            added += 1
    if added == 0:
        parts.append("- غير منصوص عليه")

    parts.append("")
    parts.append("خلاصة تنفيذية (3 نقاط)")
    summary_src = (defs[:2] + evs[:3] + comms[:2])
    added = 0
    for p in summary_src:
        if added >= 3:
            break
        if _append_unique_packet(p):
            added += 1
    while added < 3:
        parts.append("- غير منصوص عليه")
        added += 1

    return "\n".join(parts).strip(), citations

