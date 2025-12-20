"""Unit tests for deterministic sentence span extraction."""

from apps.api.ingest.sentence_spans import sentence_spans, span_text


def test_sentence_spans_basic_offsets():
    text = "مرحبا. كيف الحال؟\nسطر ثالث!"
    spans = sentence_spans(text)
    assert len(spans) >= 3

    s0 = span_text(text, spans[0])
    assert s0.endswith(".")

    s1 = span_text(text, spans[1])
    assert "؟" in s1


def test_sentence_spans_deterministic():
    text = "أولاً: تعريف. ثانياً: مثال."
    a = sentence_spans(text)
    b = sentence_spans(text)
    assert [(s.start, s.end) for s in a] == [(s.start, s.end) for s in b]
