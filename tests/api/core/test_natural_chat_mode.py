from __future__ import annotations


def test_natural_chat_mode_does_not_prepend_reasoning_block() -> None:
    """
    Natural chat mode should keep prose flowing (no Muḥāsibī methodology block).

    This is a style constraint only; safety is still enforced by guardrails.
    """

    from apps.api.core.muhasibi_interpret import build_final_response
    from apps.api.core.muhasibi_state_machine import MuhasibiMiddleware, StateContext
    from apps.api.core.muhasibi_reasoning import REASONING_START

    middleware = MuhasibiMiddleware()
    ctx = StateContext(question="اختبار", mode="natural_chat")
    ctx.answer_ar = "جواب طبيعي."
    ctx.listen_summary_ar = "ملخص"
    ctx.path_plan_ar = ["x"]
    ctx.not_found = False

    resp = build_final_response(middleware, ctx)
    assert REASONING_START not in resp.answer_ar

