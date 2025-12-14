from __future__ import annotations

from apps.api.core.muhasibi_reasoning import (
    REASONING_END,
    REASONING_START,
    build_reasoning_trace,
    render_reasoning_block,
)


def test_reasoning_trace_renders_markers_and_chain() -> None:
    trace = build_reasoning_trace(
        question="أنا أغضب بسرعة. كيف أطبق ضبط الانفعالات؟",
        detected_entities=[{"type": "sub_value", "id": "SV999", "name_ar": "ضبط الانفعالات"}],
        evidence_packets=[{"chunk_id": "CH_1"}, {"chunk_id": "CH_2"}],
        intent={"intent_type": "guidance"},
        difficulty="medium",
    )
    block = render_reasoning_block(trace)
    assert REASONING_START in block
    assert REASONING_END in block
    assert "مثير" in block and "عادة" in block
    assert "CH_1" in block


def test_reasoning_trace_handles_empty_inputs() -> None:
    trace = build_reasoning_trace(
        question="",
        detected_entities=[],
        evidence_packets=[],
        intent=None,
        difficulty=None,
    )
    block = render_reasoning_block(trace)
    assert REASONING_START in block
    assert REASONING_END in block
    # Should not claim using evidence when none exist
    assert "عدد الأدلة المسترجعة" in block


def test_reasoning_trace_does_not_crash_on_none_like_inputs() -> None:
    # type: ignore[arg-type] - defensive test
    trace = build_reasoning_trace(question=None, detected_entities=None, evidence_packets=None)
    block = render_reasoning_block(trace)
    assert REASONING_START in block
    assert REASONING_END in block

