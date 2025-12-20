from __future__ import annotations

import pytest


@pytest.mark.asyncio
async def test_runtime_contract_gate_fail_closed_when_unmet() -> None:
    """
    If contract is unmet and repair cannot run, runtime must fail-closed.
    """

    from apps.api.core.contract_gate import apply_runtime_contract_gate
    from apps.api.core.muhasibi_state_machine import MuhasibiMiddleware, StateContext
    from apps.api.core.schemas import Confidence

    mw = MuhasibiMiddleware(entity_resolver=None, retriever=None, llm_client=None, guardrails=None)
    ctx = StateContext(question="ما العلاقة بين ركيزتين؟", language="ar", mode="answer")
    ctx.normalized_question = "ما العلاقة بين ركيزتين؟"
    ctx.detected_entities = []
    ctx.deep_mode = True
    ctx.not_found = False
    ctx.answer_ar = "الربط بين الركائز (مع سبب الربط)\n- (لا توجد روابط دلالية مُبرَّرة ضمن الأدلة المسترجعة)"
    ctx.citations = []
    # Runtime contracts are enforced only when explicit requirements are present.
    ctx.answer_requirements = {"must_include": {"sections": ["الربط بين الركائز (مع سبب الربط)"]}}

    await apply_runtime_contract_gate(middleware=mw, ctx=ctx, enable_repair=False)

    assert ctx.not_found is True
    assert "عقد السؤال" in (ctx.answer_ar or "")
    assert ctx.citations == []
    assert ctx.confidence == Confidence.LOW

