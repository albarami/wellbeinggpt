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

    # LLM interpret (mode-aware)
    if self.llm_client and ctx.evidence_packets:
        result = await self.llm_client.interpret(
            question=ctx.question,
            evidence_packets=ctx.evidence_packets,
            detected_entities=ctx.detected_entities,
            mode=ctx.mode,
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
    if self.guardrails:
        try:
            result = self.guardrails.validate(
                answer_ar=ctx.answer_ar,
                citations=[c.model_dump() for c in ctx.citations],
                evidence_packets=ctx.evidence_packets,
                not_found=ctx.not_found,
            )
            if not result.passed:
                ctx.not_found = True
                ctx.answer_ar = "لا يوجد في البيانات الحالية ما يدعم الإجابة على هذا السؤال."
                ctx.citations = []
                ctx.confidence = Confidence.LOW
        except Exception:
            ctx.not_found = True
            ctx.answer_ar = "لا يوجد في البيانات الحالية ما يدعم الإجابة على هذا السؤال."
            ctx.citations = []
            ctx.confidence = Confidence.LOW

    # Entities: from detected entities (deterministic)
    ctx.entities = [
        EntityRef(
            type=EntityType(e["type"]),
            id=e["id"],
            name_ar=e["name_ar"],
        )
        for e in ctx.detected_entities
    ]


async def run_reflect(self, ctx) -> None:
    """
    Run REFLECT logic (currently minimal).
    """
    if ctx.not_found:
        ctx.reflection_added = False
        return
    # Reflection is intentionally conservative; we keep placeholder behavior.
    ctx.reflection_added = True


def build_final_response(self, ctx) -> FinalResponse:
    """Build FinalResponse from context."""
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


