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


def _is_system_limits_policy_intent(normalized_question: str) -> bool:
    """
    Detect system/framework policy questions.
    
    These ask about the system's own rules, methodology, or boundaries - not about
    the framework content itself. They don't require entity resolution.
    
    Examples:
    - "ما حدود الربط بين الركائز غير المنصوص عليها؟"
    - "ما لا ينص عليه الإطار؟"
    - "حدود الاستدلال في النظام"
    """
    q = normalized_question or ""
    markers = [
        "حدود الربط",
        "غير منصوص",
        "غير المنصوص",
        "ما لا ينص عليه",
        "حدود الاستدلال",
        "حدود النظام",
        "منهجية الربط",
        "ضوابط الربط",
        "كيف يربط النظام",
        "ما لا يغطيه الإطار",
    ]
    return any(m in q for m in markers)


def _is_guidance_framework_chat_intent(normalized_question: str, mode: str) -> bool:
    """
    Detect broad guidance/coaching questions in natural_chat mode.
    
    These are existential or self-improvement questions without specific entity anchors.
    They should be answered using pillar definitions + bridge notes.
    
    Examples:
    - "أشعر بفقدان المعنى في حياتي، ماذا أفعل؟"
    - "أريد أن أكون شخصًا أفضل، من أين أبدأ؟"
    
    Note: markers must be in NORMALIZED form (no hamza, no shadda, ى->ي).
    """
    if mode != "natural_chat":
        return False
    
    q = normalized_question or ""
    
    # Markers in normalized form (matching normalize_for_matching output)
    markers = [
        # Meaning/purpose questions
        "فقدان المعني",      # فقدان المعنى
        "المعني في حياتي",   # المعنى في حياتي
        # Self-improvement
        "اريد ان اكون افضل", # أريد أن أكون أفضل
        "اريد ان اكون شخصا", # أريد أن أكون شخصًا
        "اكون شخصا افضل",   # أكون شخصًا أفضل
        # Where to start
        "من اين ابدا",       # من أين أبدأ
        "كيف ابدا",          # كيف أبدأ
        # What to do
        "ماذا افعل",         # ماذا أفعل
        "لا اعرف ماذا افعل", # لا أعرف ماذا أفعل
        # Feeling lost
        "اشعر بالضياع",      # أشعر بالضياع
        # Need guidance
        "احتاج توجيه",       # أحتاج توجيه
        "احتاج ارشاد",       # أحتاج إرشاد
        # Self-development
        "كيف اتحسن",         # كيف أتحسن
        "كيف اطور نفسي",     # كيف أطور نفسي
        "كيف اغير حياتي",    # كيف أغير حياتي
        # Better life
        "اريد حياة افضل",    # أريد حياة أفضل
        "حياة افضل",         # حياة أفضل
    ]
    return any(m in q for m in markers)


def _is_global_synthesis_intent(normalized_question: str) -> bool:
    """
    Detect global synthesis/framework-wide questions.
    
    These ask about the framework's overall perspective, structure, or impact.
    They require seed retrieval and should bypass lexical relevance gating.
    
    Examples:
    - "ما المنظور الكلي للإطار؟"
    - "كيف تُسهم الحياة الطيبة في ازدهار الإنسان والمجتمع؟"
    - "ما الذي يميّز هذا الإطار؟"
    
    Note: markers must be in NORMALIZED form (no hamza, no shadda, ى->ي).
    """
    q = normalized_question or ""
    
    # Markers in normalized form
    synthesis_markers = [
        # Overall perspective
        "المنظور الكلي",       # المنظور الكلي
        "الرويه الشامله",      # الرؤية الشاملة
        "الرويه الكليه",       # الرؤية الكلية
        "النظره الكليه",       # النظرة الكلية
        "النظره الشامله",      # النظرة الشاملة
        # Framework uniqueness
        "يميز هذا الاطار",     # يميّز هذا الإطار
        "ما يميز الاطار",      # ما يميز الإطار
        "ما الذي يميز",        # ما الذي يميز
        # Framework contribution
        "كيف يسهم الاطار",     # كيف يُسهم الإطار
        "كيف تسهم",            # كيف تُسهم
        "ازدهار الانسان",      # ازدهار الإنسان
        "ازدهار المجتمع",      # ازدهار المجتمع
        "الحياه الطيبه",       # الحياة الطيبة
        # Framework overview
        "فلسفه الاطار",        # فلسفة الإطار
        "هدف الاطار",          # هدف الإطار
        "غايه الاطار",         # غاية الإطار
        "مقاصد الاطار",        # مقاصد الإطار
        # Integration questions
        "ترابط الركايز",       # ترابط الركائز
        "تكامل الركايز",       # تكامل الركائز
        "العلاقه بين الركايز", # العلاقة بين الركائز
        # Framework essence
        "جوهر الاطار",         # جوهر الإطار
        "ماهيه الاطار",        # ماهية الإطار
        "روح الاطار",          # روح الإطار
    ]
    
    return any(m in q for m in synthesis_markers)


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

    # External psychology frameworks / modern clinical constructs (not in values framework corpus)
    if any(k in q for k in ["ماسلو", "هرم ماسلو", "frankl", "فرانكل", "logotherapy", "cognitive behavioral", "cbt", "dsm", "icd"]):
        return True, "إطار/مدرسة نفسية خارج محتوى الإطار"
    if ("العلاج" in q) and any(k in q for k in ["المعرفي", "السلوكي", "السريري", "دوائي"]):
        return True, "سؤال علاجي/سريري خارج نطاق الإطار"

    return False, ""


def _looks_like_deep_question_ar(normalized_question: str) -> bool:
    """
    Deterministic deep-question heuristic.

    Reason: deep mode triggers a scholar-format answer with depth expansion.
    """
    q = normalized_question or ""
    markers = [
        "ما العلاقة",
        "العلاقة بين",
        "اربط بين",
        "ربط بين",
        "قارن",
        "المقارنة",
        "ما الفرق",
        "التعارض",
        "تعارض",
        "كيف نجمع",
        "التوفيق",
        "ترجيح",
        "موقف",
        "حالة",
        "سيناريو",
        "مثال",
        "تطبيق",
        # Gold QA lift: route common "explain/apply/why/how" questions into (light) deep mode.
        "عرّف",
        "عرف",
        "ما هي",
        "اشرح",
        "وضح",
        "بيّن",
        "بين",
        "اذكر",
        "قدّم",
        "قدم",
        "لماذا",
        "كيف",
        "فوائد",
        "خطوات",
        "كيف أطبق",
        "كيف يمكن",
        # Network/path intents require deep mode for graph edge selection.
        "شبكة",
        "ابنِ شبكة",
        "ابن شبكة",
        "اربطها",
        "تربطها",
        "مسار",
        "ثلاث ركائز",
        "اختر قيمة",
    ]
    return any(m in q for m in markers)


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

    # PRIORITY INTENT DETECTION (must run BEFORE LLM classifier)
    # Reason: These special intents require deterministic handling that bypasses
    # normal entity matching and relevance gating. LLM might classify them incorrectly.
    ctx.intent = None
    qn = ctx.normalized_question
    mode = getattr(ctx, "mode", "answer")
    
    # SYSTEM_LIMITS_POLICY: framework policy/methodology questions
    # These ask about the system's own rules, not framework content.
    if _is_system_limits_policy_intent(qn):
        ctx.intent = {
            "intent_type": "system_limits_policy",
            "is_in_scope": True,
            "confidence": 0.95,
            "target_entity_type": None,
            "target_entity_name_ar": None,
            "notes_ar": "سؤال عن منهجية/حدود النظام - لا يتطلب كيانات",
            "suggested_queries_ar": [],
            "required_clarification_question_ar": None,
            "bypass_relevance_gate": True,
        }
    # GUIDANCE_FRAMEWORK_CHAT: broad guidance in natural_chat mode
    # These are existential/coaching questions that use pillar seeds.
    elif _is_guidance_framework_chat_intent(qn, mode):
        ctx.intent = {
            "intent_type": "guidance_framework_chat",
            "is_in_scope": True,
            "confidence": 0.9,
            "target_entity_type": None,
            "target_entity_name_ar": None,
            "notes_ar": "سؤال إرشادي عام - يحتاج بذور الركائز",
            "suggested_queries_ar": [],
            "required_clarification_question_ar": None,
            "requires_seed_retrieval": True,
        }
    # GLOBAL_SYNTHESIS: broad framework synthesis questions
    # These ask about the framework's overall structure, perspective, or impact.
    elif _is_global_synthesis_intent(qn):
        ctx.intent = {
            "intent_type": "global_synthesis",
            "is_in_scope": True,
            "confidence": 0.9,
            "target_entity_type": None,
            "target_entity_name_ar": None,
            "notes_ar": "سؤال تركيبي كلي عن الإطار",
            "suggested_queries_ar": [],
            "required_clarification_question_ar": None,
            "requires_seed_retrieval": True,
            "bypass_relevance_gate": True,
        }

    # Intent classification (meaning-first) — optional and safe.
    # Only call LLM if we haven't detected a priority intent above.
    if (
        ctx.intent is None
        and getattr(ctx, "language", "ar") == "ar"
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

    # For network/path intents with no detected entities, find a hub entity from the DB.
    # Reason: "اختر قيمة محورية واحدة..." requires a grounded central entity.
    qn = ctx.normalized_question
    is_network_path_intent = any(
        k in qn for k in ["شبكة", "ابنِ شبكة", "ابن شبكة", "اربطها", "ثلاث ركائز", "مسار", "خطوة بخطوة"]
    )
    if is_network_path_intent and not ctx.detected_entities:
        try:
            # Import here to avoid circular imports
            from apps.api.core.scholar_reasoning_edge_fallback import find_hub_entity
            from apps.api.core.database import get_session
            
            # Get session from retriever if available
            session = getattr(getattr(self, "retriever", None), "_session", None)
            if session:
                hub = await find_hub_entity(session=session, min_distinct_pillars=3)
                if hub:
                    hub_type, hub_id = hub
                    # Look up the entity name
                    from sqlalchemy import text
                    name_row = None
                    if hub_type == "core_value":
                        name_row = (await session.execute(
                            text("SELECT name_ar FROM core_value WHERE id = :id LIMIT 1"),
                            {"id": hub_id}
                        )).fetchone()
                    elif hub_type == "sub_value":
                        name_row = (await session.execute(
                            text("SELECT name_ar FROM sub_value WHERE id = :id LIMIT 1"),
                            {"id": hub_id}
                        )).fetchone()
                    elif hub_type == "pillar":
                        name_row = (await session.execute(
                            text("SELECT name_ar FROM pillar WHERE id = :id LIMIT 1"),
                            {"id": hub_id}
                        )).fetchone()
                    
                    name_ar = str(name_row.name_ar) if name_row and name_row.name_ar else f"{hub_type}:{hub_id}"
                    ctx.detected_entities = [{
                        "type": hub_type,
                        "id": hub_id,
                        "name_ar": name_ar,
                        "confidence": 0.7,
                        "source": "hub_entity_fallback",
                    }]
        except Exception:
            # Fail open - continue without hub entity
            pass

    # Listen summary
    if ctx.detected_entities:
        entity_names = ", ".join(e["name_ar"] for e in ctx.detected_entities[:3])
        ctx.listen_summary_ar = f"السؤال عن: {entity_names}"
    else:
        ctx.listen_summary_ar = f"سؤال عام: {ctx.question[:100]}"

    # Deep-mode flag (deterministic).
    # Note: list/structure intents are handled separately and should not be treated as deep mode.
    try:
        intent_type = str((ctx.intent or {}).get("intent_type") or "")
        if intent_type in {"list_pillars", "list_core_values_in_pillar", "list_sub_values_in_core_value"}:
            ctx.deep_mode = False
        else:
            ctx.deep_mode = _looks_like_deep_question_ar(ctx.normalized_question)
    except Exception:
        ctx.deep_mode = _looks_like_deep_question_ar(ctx.normalized_question)


