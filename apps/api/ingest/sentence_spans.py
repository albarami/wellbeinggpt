"""Deterministic sentence span extraction.

Used for:
- canonical citation spans (chunk sentence offsets)
- eval dataset generation (gold_supporting_spans)

Design:
- Pure Python, deterministic.
- Offsets are character offsets into the *stored chunk text*.
"""

from __future__ import annotations

from dataclasses import dataclass


_SENTENCE_END = {".", "!", "؟", "؛", "\n"}


@dataclass(frozen=True)
class TextSpan:
    start: int
    end: int


def sentence_spans(text: str, *, max_spans: int = 64) -> list[TextSpan]:
    """Return sentence-like spans with stable offsets.

    Notes:
    - This is intentionally conservative; it prefers shorter, well-bounded spans.
    - Trims leading/trailing whitespace from each span.

    Args:
        text: Input text.
        max_spans: Hard cap to avoid pathological text.

    Returns:
        List of TextSpan with [start, end) offsets.
    """
    if not text:
        return []

    spans: list[TextSpan] = []
    start = 0
    n = len(text)

    def _emit(raw_start: int, raw_end: int) -> None:
        nonlocal spans
        if raw_end <= raw_start:
            return

        # Trim whitespace while keeping offsets correct.
        s = raw_start
        e = raw_end
        while s < e and text[s].isspace():
            s += 1
        while e > s and text[e - 1].isspace():
            e -= 1
        if e <= s:
            return
        spans.append(TextSpan(start=s, end=e))

    for i, ch in enumerate(text):
        if ch in _SENTENCE_END:
            # Include delimiter in the span for better entailment checks.
            _emit(start, i + 1)
            start = i + 1
            if len(spans) >= max_spans:
                break

    if len(spans) < max_spans and start < n:
        _emit(start, n)

    # Dedupe identical spans (rare but can happen with repeated whitespace splits)
    seen: set[tuple[int, int]] = set()
    out: list[TextSpan] = []
    for sp in spans:
        key = (sp.start, sp.end)
        if key in seen:
            continue
        seen.add(key)
        out.append(sp)

    return out


def span_text(text: str, span: TextSpan) -> str:
    return text[span.start : span.end]
