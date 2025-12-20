"""
Muḥāsibī ACCOUNT helpers.

Split out of `muhasibi_state_machine.py` to keep files under 500 lines.
"""

from __future__ import annotations

from apps.api.retrieve.normalize_ar import extract_arabic_words, normalize_for_matching


def _contains_any(text_norm: str, needles: list[str]) -> bool:
    for n in needles:
        if normalize_for_matching(n) in text_norm:
            return True
    return False


def _token_norm_set(text: str) -> set[str]:
    toks = extract_arabic_words(text or "")
    return {normalize_for_matching(t) for t in toks if len(t) >= 2}


def _is_fiqh_ruling_question(question_norm: str) -> bool:
    """
    Heuristic for fiqh rulings (out-of-scope for this framework unless explicitly ingested).
    """
    fiqh_markers = [
        "ما حكم",
        "حكم",
        "يجوز",
        "لا يجوز",
        "حلال",
        "حرام",
        "مكروه",
        "سنة",
        "فرض",
        "واجب",
        "مندوب",
        "مباح",
        "بدعة",
    ]
    worship_terms = [
        "صيام",
        "صوم",
        "الجمعة",
        "صلاة",
        "زكاة",
        "حج",
        "عمرة",
    ]
    return _contains_any(question_norm, fiqh_markers) and _contains_any(question_norm, worship_terms)


def apply_question_evidence_relevance_gate(ctx) -> None:
    """
    If the question looks out-of-scope (no entities) and the retrieved evidence
    doesn't contain any meaningful question terms, fail closed.

    This prevents "answering" unrelated fiqh/medical questions using random
    wellbeing evidence.
    """
    packets = getattr(ctx, "evidence_packets", None) or []
    if not packets:
        return

    question = getattr(ctx, "question", "") or ""
    q_norm = normalize_for_matching(question)

    # If this is a structure/list intent, do NOT apply lexical relevance gating.
    # Reason: structure answers are backed by heading/structure chunks which may not
    # share many surface tokens with the user's phrasing, but are still correct + citeable.
    try:
        intent = getattr(ctx, "intent", None) or {}
        intent_type = (intent.get("intent_type") or "").strip()
        if intent_type in {
            "list_pillars",
            "list_core_values_in_pillar",
            "list_sub_values_in_core_value",
        }:
            return
    except Exception:
        pass

    # BYPASS: system_limits_policy intent (policy/methodology questions)
    # Reason: These ask about the system's own rules, not framework content.
    # They don't require entity matches - we answer from system policy.
    try:
        intent = getattr(ctx, "intent", None) or {}
        intent_type = (intent.get("intent_type") or "").strip()
        if intent_type == "system_limits_policy":
            return
        if intent.get("bypass_relevance_gate"):
            return
    except Exception:
        pass

    # BYPASS: guidance_framework_chat intent (broad guidance questions)
    # Reason: These are existential/coaching questions that use pillar seeds.
    # They don't require entity matches - we answer from pillar definitions + bridges.
    try:
        intent = getattr(ctx, "intent", None) or {}
        intent_type = (intent.get("intent_type") or "").strip()
        if intent_type == "guidance_framework_chat":
            return
        if intent.get("requires_seed_retrieval"):
            return
    except Exception:
        pass

    # BYPASS: global_synthesis, cross_pillar, network_build intents
    # Reason: These require broad, cross-pillar reasoning and may not have specific entities.
    try:
        intent = getattr(ctx, "intent", None) or {}
        intent_type = (intent.get("intent_type") or "").strip()
        if intent_type in {
            "global_synthesis",
            "cross_pillar",
            "network_build",
            "compare",
            "world_model",
        }:
            return
    except Exception:
        pass

    # BYPASS: natural_chat mode with seed floor met
    # Reason: natural_chat guidance questions can be answered from pillar definitions
    # even without specific entity matches. Check if we have seed evidence.
    try:
        mode = getattr(ctx, "mode", "") or ""
        if mode == "natural_chat" and len(packets) >= 5:
            # Check if we have pillar definition seeds (at least 3)
            pillar_seeds = [
                p for p in packets
                if p.get("entity_type") == "pillar" or "ركيزة" in (p.get("text_ar") or "")
            ]
            if len(pillar_seeds) >= 3:
                return
    except Exception:
        pass

    # If we have high-confidence detected framework entities, do NOT apply lexical relevance gating.
    # Reason: retrieval is anchored to in-corpus entities; lexical mismatch would create false
    # refusals for valid guidance questions (e.g., balancing work/rest).
    #
    # IMPORTANT: low-confidence matches (or noisy matches) must not bypass the out-of-scope gate,
    # otherwise we will \"answer\" unrelated questions using random wellbeing evidence.
    try:
        det = getattr(ctx, "detected_entities", None) or []
        high_conf = [d for d in det if float(d.get("confidence") or 0.0) >= 0.75]
        if len(high_conf) > 0:
            return
    except Exception:
        pass

    # Heuristic fallback: if user explicitly asked for listing pillars (ركائز/الخمس) and
    # we already detected the 5 pillars, skip relevance gating.
    try:
        det = getattr(ctx, "detected_entities", None) or []
        if (
            len(det) == 5
            and all((d.get("type") == "pillar") for d in det)
            and ("ركائز" in q_norm or "اركان" in q_norm)
            and ("الخمس" in q_norm or "خمسة" in q_norm or "5" in q_norm)
        ):
            return
    except Exception:
        pass

    # Special-case: fiqh ruling questions should be refused.
    # Reason: The wellbeing framework is not a general fiqh engine.
    if _is_fiqh_ruling_question(q_norm):
        ctx.not_found = True
        ctx.citation_valid = False
        ctx.account_issues.append("السؤال فقهي/حُكمي (فتوى) خارج نطاق النظام")

        # Provide a best in-scope reframing suggestion (no answer, no evidence needed).
        # This is only a suggested question, not a claim.
        if ("صيام" in q_norm) or ("صوم" in q_norm):
            ctx.refusal_suggestion_ar = (
                "بديل داخل النطاق: ما أثر الصيام كعبادة على تزكية النفس والطاعة ضمن إطار الحياة الطيبة؟"
            )
        else:
            ctx.refusal_suggestion_ar = (
                "بديل داخل النطاق: كيف يرتبط هذا الموضوع بقيم الطاعة/العبادة/التزكية ضمن إطار الحياة الطيبة؟"
            )
        return

    # Extract meaningful question terms (fallback relevance)
    q_terms = [t for t in extract_arabic_words(question) if len(t) >= 3]
    if not q_terms:
        return

    # Combine top evidence texts
    combined = " ".join([(p.get("text_ar") or "") for p in packets[:12]])
    combined_norm = normalize_for_matching(combined)

    matched = 0
    for t in q_terms[:12]:
        if normalize_for_matching(t) in combined_norm:
            matched += 1

    # If nothing matches, treat as out-of-scope.
    if matched == 0:
        ctx.not_found = True
        ctx.citation_valid = False
        ctx.account_issues.append("السؤال خارج نطاق البيانات المتاحة أو لا توجد صلة كافية بالأدلة")


