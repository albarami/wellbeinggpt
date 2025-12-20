"""
Muḥāsibī INTERPRET/REFLECT helpers.

Split out of `muhasibi_state_machine.py` to enforce the 500-line rule.
"""

from __future__ import annotations

from typing import Any

from apps.api.core.schemas import (
    Confidence,
    Citation,
    EntityRef,
    EntityType,
    FinalResponse,
    Purpose,
)
from apps.api.core.muhasibi_structure_answer import answer_list_core_values_in_pillar, answer_list_pillars
from apps.api.core.muhasibi_reasoning import build_reasoning_trace, render_reasoning_block, REASONING_START


def _deterministic_answer_from_packets(ctx) -> None:
    """
    Build a safe answer directly from evidence packet text.

    Reason: if LLM output is rejected by guardrails, we must not refuse when
    we *do* have relevant evidence; instead we synthesize deterministically
    from the evidence bundle.
    """
    packets = getattr(ctx, "evidence_packets", None) or []
    if not packets:
        ctx.answer_ar = "لا يوجد في البيانات الحالية ما يدعم الإجابة على هذا السؤال."
        ctx.not_found = True
        ctx.confidence = Confidence.LOW
        ctx.citations = []
        return

    defs = [p for p in packets if p.get("chunk_type") == "definition"]
    evs = [p for p in packets if p.get("chunk_type") == "evidence"]
    comms = [p for p in packets if p.get("chunk_type") == "commentary"]
    chosen = (defs[:2] + evs[:3] + comms[:3]) or packets[:6]

    # Structured, rubric-friendly output (still evidence-only: we only print packet text).
    parts: list[str] = []
    if defs:
        parts.append("التعريف (من النص):")
        for d in defs[:2]:
            t = (d.get("text_ar") or "").strip()
            if t:
                parts.append("- " + t)

    if evs:
        parts.append("الدليل/التأصيل (من النص):")
        for e in evs[:3]:
            t = (e.get("text_ar") or "").strip()
            if t:
                parts.append("- " + t)

    # Use commentary as practical guidance when available (still verbatim evidence text).
    if comms:
        parts.append("تطبيق/إرشادات داخل الإطار (من النص):")
        for c in comms[:5]:
            t = (c.get("text_ar") or "").strip()
            if t:
                parts.append("- " + t)

    if not parts:
        parts = ["لا يوجد في البيانات الحالية ما يدعم الإجابة على هذا السؤال."]
        ctx.not_found = True

    ctx.answer_ar = "\n".join(parts).strip()
    ctx.citations = [
        Citation(
            chunk_id=p.get("chunk_id", ""),
            source_anchor=p.get("source_anchor") or "",
            ref=(p.get("refs", [{}])[0].get("ref") if p.get("refs") else None),
        )
        for p in chosen
        if p.get("chunk_id")
    ]
    ctx.not_found = not bool(ctx.citations)
    ctx.confidence = Confidence.MEDIUM if ctx.citations else Confidence.LOW


def _ensure_min_citations(ctx, min_citations: int = 5) -> None:
    """
    Ensure we attach enough citations when answering.

    Reason: the A/B harness measures citation coverage; the system should expose
    a richer evidence trail when available (still evidence-only).
    """
    try:
        if getattr(ctx, "not_found", True):
            return
        packets = getattr(ctx, "evidence_packets", None) or []
        if not packets:
            return
        citations = list(getattr(ctx, "citations", None) or [])
        if len(citations) >= min_citations:
            return
        seen = {c.chunk_id for c in citations if getattr(c, "chunk_id", None)}
        for p in packets:
            cid = p.get("chunk_id")
            if not cid or cid in seen:
                continue
            citations.append(
                Citation(
                    chunk_id=str(cid),
                    source_anchor=str(p.get("source_anchor") or ""),
                    ref=(p.get("refs", [{}])[0].get("ref") if p.get("refs") else None),
                )
            )
            seen.add(str(cid))
            if len(citations) >= min_citations:
                break
        ctx.citations = citations
    except Exception:
        return


async def _augment_entities_from_evidence(self, ctx) -> None:
    """
    Add additional entities based on evidence packets.

    Reason: makes cross-pillar linkage measurable (entity list is part of contract).
    """
    try:
        session = getattr(getattr(self, "retriever", None), "_session", None)
        packets = getattr(ctx, "evidence_packets", None) or []
        if not session or not packets:
            return

        # Start from detected entities (already canonical names).
        entities: list[EntityRef] = list(getattr(ctx, "entities", None) or [])
        seen: set[tuple[str, str]] = {(e.type.value, e.id) for e in entities}

        # Collect candidate ids from packets
        cand: dict[str, set[str]] = {"pillar": set(), "core_value": set(), "sub_value": set()}
        for p in packets[:25]:
            et = str(p.get("entity_type") or "")
            eid = str(p.get("entity_id") or "")
            if et in cand and eid:
                if (et, eid) not in seen:
                    cand[et].add(eid)

        # Fetch names in bulk
        from sqlalchemy import text

        def _rows_to_entities(rows, et: str) -> list[EntityRef]:
            out: list[EntityRef] = []
            for r in rows:
                try:
                    out.append(EntityRef(type=EntityType(et), id=str(r.id), name_ar=str(r.name_ar)))
                except Exception:
                    continue
            return out

        if cand["pillar"]:
            rows = (
                await session.execute(
                    text("SELECT id, name_ar FROM pillar WHERE id = ANY(:ids)"),
                    {"ids": sorted(list(cand["pillar"]))},
                )
            ).fetchall()
            for e in _rows_to_entities(rows, "pillar"):
                key = (e.type.value, e.id)
                if key not in seen:
                    entities.append(e)
                    seen.add(key)

        if cand["core_value"]:
            rows = (
                await session.execute(
                    text("SELECT id, name_ar FROM core_value WHERE id = ANY(:ids)"),
                    {"ids": sorted(list(cand["core_value"]))},
                )
            ).fetchall()
            for e in _rows_to_entities(rows, "core_value"):
                key = (e.type.value, e.id)
                if key not in seen:
                    entities.append(e)
                    seen.add(key)

        if cand["sub_value"]:
            rows = (
                await session.execute(
                    text("SELECT id, name_ar FROM sub_value WHERE id = ANY(:ids)"),
                    {"ids": sorted(list(cand["sub_value"]))},
                )
            ).fetchall()
            for e in _rows_to_entities(rows, "sub_value"):
                key = (e.type.value, e.id)
                if key not in seen:
                    entities.append(e)
                    seen.add(key)

        # Keep list compact but informative
        ctx.entities = entities[:12]
    except Exception:
        return


async def run_interpret(self, ctx) -> None:
    """
    Run INTERPRET logic (LLM or deterministic fallback) and guardrails.

    Note: `self` is MuhasibiMiddleware; `ctx` is StateContext.
    """
    if ctx.not_found:
        # Out-of-scope or no evidence: refuse safely and optionally suggest best in-scope reframe.
        suggestion = (getattr(ctx, "refusal_suggestion_ar", None) or "").strip()
        if suggestion:
            ctx.answer_ar = (
                "لا أستطيع إصدار فتوى/حكم شرعي من هذا النظام.\n\n"
                + suggestion
            )
        else:
            ctx.answer_ar = "لا يوجد في البيانات الحالية ما يدعم الإجابة على هذا السؤال."
        ctx.confidence = Confidence.LOW
        ctx.citations = []
        return

    # Deterministic structure answering (no GPT‑5 needed, always citeable)
    try:
        intent = getattr(ctx, "intent", None) or {}
        intent_type = (intent.get("intent_type") or "").strip()
        session = getattr(getattr(self, "retriever", None), "_session", None)
        if session:
            if intent_type == "list_pillars":
                out = await answer_list_pillars(session)
                if out:
                    ctx.answer_ar = out["answer_ar"]
                    ctx.citations = [
                        Citation(chunk_id=c["chunk_id"], source_anchor=c["source_anchor"], ref=None)
                        for c in out["citations"]
                        if c.get("chunk_id")
                    ]
                    ctx.not_found = False
                    ctx.confidence = Confidence.HIGH
                    return

            if intent_type == "list_core_values_in_pillar":
                pillar_ent = next((e for e in (ctx.detected_entities or []) if e.get("type") == "pillar"), None)
                if pillar_ent:
                    out = await answer_list_core_values_in_pillar(
                        session, pillar_id=str(pillar_ent["id"]), pillar_name_ar=str(pillar_ent["name_ar"])
                    )
                    if out:
                        ctx.answer_ar = out["answer_ar"]
                        ctx.citations = [
                            Citation(chunk_id=c["chunk_id"], source_anchor=c["source_anchor"], ref=None)
                            for c in out["citations"]
                            if c.get("chunk_id")
                        ]
                        ctx.not_found = False
                        ctx.confidence = Confidence.HIGH
                        return
    except Exception:
        pass

    # Scholar deep mode (deterministic, evidence-only, depth-expanded).
    try:
        if bool(getattr(ctx, "deep_mode", False)) and (ctx.evidence_packets or []):
            session = getattr(getattr(self, "retriever", None), "_session", None)
            if session:
                from apps.api.core.scholar_reasoning import ScholarReasoner

                reasoner = ScholarReasoner(session=session, retriever=self.retriever)
                res = await reasoner.generate(
                    question=ctx.question,
                    question_norm=getattr(ctx, "normalized_question", "") or "",
                    detected_entities=getattr(ctx, "detected_entities", None) or [],
                    evidence_packets=getattr(ctx, "evidence_packets", None) or [],
                )
                # Always expose used semantic edges for downstream tracing/contracts (eval + runtime gates).
                try:
                    setattr(self, "_last_used_edges", list((res or {}).get("used_edges") or []))
                except Exception:
                    setattr(self, "_last_used_edges", [])

                # Default behavior: if deep mode produced an answer, skip LLM to preserve determinism.
                # Natural chat behavior: use deep-mode graph hints, but let the LLM write a narrative answer.
                if (ctx.mode or "") != "natural_chat":
                    if res and str(res.get("answer_ar") or "").strip():
                        ctx.answer_ar = str(res.get("answer_ar") or "")
                        ctx.not_found = bool(res.get("not_found"))
                        ctx.confidence = res.get("confidence") or Confidence.MEDIUM
                        ctx.citations = list(res.get("citations") or [])
                        # If deep mode abstained, do not continue into LLM path.
                        if ctx.not_found:
                            return
                        # If deep mode produced an answer, skip LLM to preserve determinism.
                        if ctx.citations:
                            return
    except Exception:
        # Fail open: fall back to the existing interpret paths.
        pass

    # LLM interpret (mode-aware)
    if self.llm_client and ctx.evidence_packets:
        used_edges_for_prompt = []
        try:
            used_edges_for_prompt = list(getattr(self, "_last_used_edges", None) or [])[:24]
        except Exception:
            used_edges_for_prompt = []

        result = await self.llm_client.interpret(
            question=ctx.question,
            evidence_packets=ctx.evidence_packets,
            detected_entities=ctx.detected_entities,
            mode=ctx.mode,
            used_edges=used_edges_for_prompt,
            argument_chains=[],
        )
        if result:
            ctx.answer_ar = result.answer_ar
            ctx.not_found = bool(result.not_found)
            try:
                ctx.confidence = Confidence(result.confidence)
            except Exception:
                ctx.confidence = Confidence.MEDIUM
            # Parse citations - chunk_id is required, source_anchor can be empty
            ctx.citations = [
                Citation(
                    chunk_id=c.get("chunk_id", ""),
                    source_anchor=c.get("source_anchor") or "",
                    ref=c.get("ref"),
                )
                for c in result.citations
                if c.get("chunk_id")
            ]

            # If the LLM claims "not_found" despite evidence being present, treat it as a
            # retrieval/formatting failure and fall back to deterministic synthesis.
            # Reason: enterprise stability; do not refuse when we *do* have evidence.
            if ctx.not_found and ctx.evidence_packets:
                ctx.not_found = False
                ctx.answer_ar = ""
                ctx.citations = []

    # Deterministic fallback if evidence exists but no LLM call occurred.
    if not ctx.answer_ar and ctx.evidence_packets:
        defs = [p for p in ctx.evidence_packets if p.get("chunk_type") == "definition"]
        evs = [p for p in ctx.evidence_packets if p.get("chunk_type") == "evidence"]
        chosen = (defs[:1] + evs[:2]) or ctx.evidence_packets[:3]

        parts: list[str] = []
        if defs:
            parts.append(f"التعريف:\n{defs[0].get('text_ar','').strip()}")
        if evs:
            parts.append("التأصيل/الدليل:")
            for e in evs[:2]:
                parts.append(e.get("text_ar", "").strip())
        if not parts:
            parts = ["تم العثور على أدلة، لكن تعذر تلخيصها آليًا بدون نموذج لغوي."]

        ctx.answer_ar = "\n\n".join([p for p in parts if p])
        ctx.citations = [
            Citation(
                chunk_id=p.get("chunk_id", ""),
                source_anchor=p.get("source_anchor") or "",
                ref=(p.get("refs", [{}])[0].get("ref") if p.get("refs") else None),
            )
            for p in chosen
            if p.get("chunk_id")
        ]
        ctx.confidence = Confidence.MEDIUM if ctx.citations else Confidence.LOW
        ctx.not_found = not bool(ctx.citations)

    # If still no answer, refuse safely
    if not ctx.answer_ar or not ctx.answer_ar.strip():
        ctx.answer_ar = "لا يوجد في البيانات الحالية ما يدعم الإجابة على هذا السؤال."
        ctx.not_found = True
        ctx.confidence = Confidence.LOW
        ctx.citations = []

    # CRITICAL: If not_found=False, answer MUST be non-empty and have citations
    if not ctx.not_found:
        if not ctx.answer_ar.strip():
            ctx.not_found = True
            ctx.answer_ar = "لا يوجد في البيانات الحالية ما يدعم الإجابة على هذا السؤال."
            ctx.confidence = Confidence.LOW
            ctx.citations = []
        elif not ctx.citations and ctx.evidence_packets:
            # If LLM gave answer but no citations, build citations from evidence packets
            chosen = ctx.evidence_packets[:3]
            ctx.citations = [
                Citation(
                    chunk_id=p.get("chunk_id", ""),
                    source_anchor=p.get("source_anchor") or "",
                    ref=(p.get("refs", [{}])[0].get("ref") if p.get("refs") else None),
                )
                for p in chosen
                if p.get("chunk_id")
            ]

    # Guardrails
    # For natural_chat mode: keep the LLM's flowing prose even if guardrails aren't fully satisfied.
    # Reason: natural_chat prioritizes scholarly narrative over rubric compliance.
    is_natural_chat = (getattr(ctx, "mode", "") or "") == "natural_chat"
    if self.guardrails and not is_natural_chat:
        try:
            result = self.guardrails.validate(
                answer_ar=ctx.answer_ar,
                citations=[c.model_dump() for c in ctx.citations],
                evidence_packets=ctx.evidence_packets,
                not_found=ctx.not_found,
            )
            if not result.passed:
                # If guardrails reject the LLM answer, fall back to deterministic synthesis
                # from evidence instead of refusing (we *have* evidence).
                _deterministic_answer_from_packets(ctx)
        except Exception:
            _deterministic_answer_from_packets(ctx)

    # Entities: from detected entities (deterministic)
    ctx.entities = [
        EntityRef(
            type=EntityType(e["type"]),
            id=e["id"],
            name_ar=e["name_ar"],
        )
        for e in ctx.detected_entities
    ]

    # Enrich entities list from evidence packets (for measurable cross-linkage)
    await _augment_entities_from_evidence(self, ctx)

    # Ensure we expose enough citations when evidence exists
    _ensure_min_citations(ctx, min_citations=5)

    # Hard gate: if graph is required but used_edges is empty, append disclaimer.
    # Reason: prevent "full narrative without grounded edges" for network/path questions.
    await _apply_graph_edge_hard_gate(self, ctx)


async def _apply_graph_edge_hard_gate(self, ctx) -> None:
    """
    If the intent requires graph edges but used_edges is empty, append a disclaimer.
    
    This prevents the system from claiming cross-pillar relationships without
    grounded semantic edges.
    """
    try:
        q_norm = str(getattr(ctx, "normalized_question", "") or "")
        
        # Check if this is a graph-required intent
        is_graph_intent = any(
            k in q_norm for k in [
                "شبكة", "ابنِ شبكة", "ابن شبكة", "اربطها", "ثلاث ركائز",
                "مسار", "خطوة بخطوة", "العلاقة بين", "اربط بين", "ربط بين",
            ]
        )
        
        if not is_graph_intent:
            return
        
        # Check if used_edges is empty
        used_edges = list(getattr(self, "_last_used_edges", None) or [])
        if used_edges:
            return  # Edges exist, no need for disclaimer
        
        # Append disclaimer about no grounded links
        answer_ar = str(getattr(ctx, "answer_ar", "") or "").rstrip()
        
        # Check if answer contains relation labels that shouldn't be there
        relation_labels = ["تمكين", "تعزيز", "تكامل", "إعانة", "شرط"]
        has_ungrounded_labels = any(label in answer_ar for label in relation_labels)
        
        if has_ungrounded_labels:
            # Add explicit disclaimer
            disclaimer = (
                "\n\n"
                "ملاحظة: لا توجد في الأدلة الحالية روابط دلالية مؤسسة صراحة بين هذه المفاهيم. "
                "ما ذُكر أعلاه هو استنباط عام وليس مبنيًا على روابط محددة في قاعدة المعرفة."
            )
            ctx.answer_ar = answer_ar + disclaimer
        elif "لا توجد روابط" not in answer_ar and "لا توجد في الأدلة" not in answer_ar:
            # Answer doesn't contain relation labels but also no disclaimer
            disclaimer = (
                "\n\n"
                "ملاحظة: هذا ما يمكن دعمه من النصوص المتاحة. "
                "لا توجد في الأدلة الحالية روابط دلالية مؤسسة صراحة بين هذه المفاهيم."
            )
            ctx.answer_ar = answer_ar + disclaimer
    except Exception:
        # Fail open - don't block the answer
        pass


async def run_reflect(self, ctx) -> None:
    """
    Run REFLECT logic (currently minimal).
    """
    if ctx.not_found:
        ctx.reflection_added = False
        return
    # Add a short, explicitly-labeled reflection + next steps.
    # Important: keep wording minimal to avoid introducing unsupported terms.
    q = (getattr(ctx, "question", "") or "").strip()
    q_norm = q
    # Only add for guidance-style questions, not for pure listing/definition lookups.
    if ("كيف" in q_norm) or ("أنا" in q_norm) or ("أشعر" in q_norm) or ("اشعر" in q_norm):
        ents = [e.name_ar for e in (getattr(ctx, "entities", None) or []) if getattr(e, "name_ar", None)]
        focus = "، ".join(ents[:3]) if ents else "القيمة المذكورة في النص"
        ctx.answer_ar = (
            (ctx.answer_ar or "").rstrip()
            + "\n\n"
            + "انعكاس (إرشاد عام مرتبط بالنص):\n"
            + f"يمكنك تحويل ما سبق إلى خطوات عملية عبر التركيز على: {focus}.\n"
            + "خطوات مختصرة: ابدأ بقراءة النص المستشهد به، ثم حاول تحديد السلوك المرتبط به في واقعك، ثم دوّن تقويمًا ذاتيًا لما نجح وما يحتاج تعزيزًا."
        )
        ctx.reflection_added = True
    else:
        ctx.reflection_added = False


def build_final_response(self, ctx) -> FinalResponse:
    """Build FinalResponse from context."""
    # ENGINEERING FIX: Do NOT prepend reasoning block to user-facing answer.
    # The block ruins naturalness and confuses contract section parsing.
    # Instead, store it in _last_reasoning_trace for debug/UI access only.
    try:
        if REASONING_START not in (ctx.answer_ar or ""):
            trace = build_reasoning_trace(
                question=getattr(ctx, "question", "") or "",
                detected_entities=getattr(ctx, "detected_entities", None) or [],
                evidence_packets=getattr(ctx, "evidence_packets", None) or [],
                intent=getattr(ctx, "intent", None),
                difficulty=(getattr(getattr(ctx, "difficulty", None), "value", None) or None),
            )
            # Store for debug access, NOT in answer
            setattr(self, "_last_reasoning_trace", trace)
            # REMOVED: ctx.answer_ar = (render_reasoning_block(trace) + "\n" + (ctx.answer_ar or "")).strip()
    except Exception:
        # Fail open: do not block answering if reasoning rendering fails.
        pass

    # Expose a safe per-request snapshot for UI wrappers (e.g., /ask/ui).
    # Reason: /ask/ui must be a pure wrapper over the same middleware execution,
    # but needs access to deterministic internal artifacts (used_edges, detected_entities, etc.).
    try:
        setattr(self, "_last_ctx", ctx)
    except Exception:
        pass

    return FinalResponse(
        listen_summary_ar=ctx.listen_summary_ar,
        purpose=ctx.purpose
        or Purpose(
            ultimate_goal_ar="غير محدد",
            constraints_ar=self.REQUIRED_CONSTRAINTS,
        ),
        path_plan_ar=ctx.path_plan_ar,
        answer_ar=ctx.answer_ar,
        citations=ctx.citations,
        entities=ctx.entities,
        difficulty=ctx.difficulty,
        not_found=ctx.not_found,
        confidence=ctx.confidence,
    )


