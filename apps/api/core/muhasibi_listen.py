"""
Muḥāsibī LISTEN helpers (Arabic meaning-first).

Split out of `muhasibi_state_machine.py` to keep files under 500 lines and
to implement intent classification cleanly.
"""

from __future__ import annotations

import os

from apps.api.retrieve.normalize_ar import extract_arabic_words, normalize_for_matching


def _is_pillar_list_intent(normalized_question: str) -> bool:
    q = normalized_question
    return ("ركائز" in q or "اركان" in q) and ("الخمس" in q or "خمسة" in q or "5" in q)


def _detect_out_of_scope_ar(normalized_question: str) -> tuple[bool, str]:
    """
    Detect out-of-scope questions deterministically (Arabic-first).

    This system is scoped to the wellbeing framework content (pillars/values/evidence within the corpus).
    Questions about medicine/drugs, specific fiqh rulings, authorship, etc. are out of scope.
    """
    q = normalized_question or ""

    # Medical / drug questions
    if ("فوائد" in q and "طبي" in q) or ("دواء" in q) or ("ادوية" in q) or ("علاج" in q):
        return True, "سؤال طبي/دوائي خارج نطاق الإطار"
    if "باراسيتامول" in q or "paracetamol" in q or "acetaminophen" in q:
        return True, "سؤال طبي/دوائي خارج نطاق الإطار"

    # Specific fiqh ruling questions (not values framework)
    if ("ما حكم" in q or "الحكم الشرعي" in q or "هل يجوز" in q or "حرمة" in q or "حلال" in q or "حرام" in q):
        return True, "سؤال فقهي/إفتائي خارج نطاق الإطار"

    # Authorship / biography / external facts
    if "من هو مؤلف" in q or "من مؤلف" in q:
        return True, "سؤال معلومات خارج نطاق الإطار"

    return False, ""


async def run_listen(self, ctx) -> None:
    """
    Populate ctx for LISTEN:
    - normalize question
    - keywords
    - entity resolution
    - (optional) intent classification via GPT‑5 (structured, no answering)
    - deterministic structure intent helpers
    """
    ctx.normalized_question = normalize_for_matching(ctx.question)
    ctx.question_keywords = extract_arabic_words(ctx.question)

    # Deterministic out-of-scope refusal (fail closed).
    oos, reason = _detect_out_of_scope_ar(ctx.normalized_question)
    if oos:
        ctx.not_found = True
        ctx.intent = {
            "intent_type": "out_of_scope",
            "is_in_scope": False,
            "confidence": 0.95,
            "target_entity_type": None,
            "target_entity_name_ar": None,
            "notes_ar": reason,
            "suggested_queries_ar": [],
            "required_clarification_question_ar": None,
        }
        ctx.listen_summary_ar = f"خارج النطاق: {reason}"
        return

    # Entity resolution
    if self.entity_resolver:
        resolved = self.entity_resolver.resolve(ctx.question)
        ctx.detected_entities = [
            {
                "type": r.entity_type.value,
                "id": str(r.entity_id),
                "name_ar": r.name_ar,
                "confidence": r.confidence,
            }
            for r in resolved
        ]

        # Deterministic structure intent: list all pillars if asked explicitly.
        try:
            if _is_pillar_list_intent(ctx.normalized_question):
                pillars = getattr(self.entity_resolver, "list_pillars", lambda: [])()
                if pillars and len(pillars) == 5:
                    ctx.detected_entities = [
                        {"type": "pillar", "id": str(p["id"]), "name_ar": p["name_ar"], "confidence": 0.8}
                        for p in pillars
                    ]
        except Exception:
            pass

    # Intent classification (meaning-first) — optional and safe.
    ctx.intent = None
    if (
        getattr(ctx, "language", "ar") == "ar"
        and self.llm_client
        and os.getenv("ENABLE_INTENT_CLASSIFIER", "1") == "1"
    ):
        try:
            intent = await self.llm_client.classify_intent_ar(
                question=ctx.question,
                detected_entities=ctx.detected_entities,
                keywords=ctx.question_keywords,
            )
            ctx.intent = intent
        except Exception:
            ctx.intent = None

    # Deterministic fallback intent (keeps behavior stable even if LLM is down)
    if not ctx.intent:
        qn = ctx.normalized_question
        # list pillars
        if _is_pillar_list_intent(qn):
            ctx.intent = {
                "intent_type": "list_pillars",
                "is_in_scope": True,
                "confidence": 0.9,
                "target_entity_type": "pillar",
                "target_entity_name_ar": None,
                "notes_ar": "طلب سرد الركائز الخمس",
                "suggested_queries_ar": [],
                "required_clarification_question_ar": None,
            }
        # list core values in a pillar (القيم الكلية في الركيزة X)
        elif ("القيم" in qn) and ("الكلية" in qn) and any(e.get("type") == "pillar" for e in ctx.detected_entities):
            ctx.intent = {
                "intent_type": "list_core_values_in_pillar",
                "is_in_scope": True,
                "confidence": 0.8,
                "target_entity_type": "pillar",
                "target_entity_name_ar": next((e.get("name_ar") for e in ctx.detected_entities if e.get("type") == "pillar"), None),
                "notes_ar": "طلب سرد القيم الكلية في ركيزة",
                "suggested_queries_ar": [],
                "required_clarification_question_ar": None,
            }

    # Listen summary
    if ctx.detected_entities:
        entity_names = ", ".join(e["name_ar"] for e in ctx.detected_entities[:3])
        ctx.listen_summary_ar = f"السؤال عن: {entity_names}"
    else:
        ctx.listen_summary_ar = f"سؤال عام: {ctx.question[:100]}"


