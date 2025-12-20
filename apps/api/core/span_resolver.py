"""
Deterministic citation span resolver for the UI (/ask/ui).

Rules (non-negotiable):
- Do not guess offsets.
- If offsets cannot be resolved reliably, return span_start/span_end = None and status=unresolved.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from apps.api.ingest.sentence_spans import sentence_spans, span_text
from apps.api.retrieve.normalize_ar import normalize_for_matching


@dataclass(frozen=True)
class SpanResolution:
    chunk_id: str
    quote: str
    span_start: Optional[int]
    span_end: Optional[int]
    status: str  # resolved|unresolved
    method: str
    score: float = 0.0


def _token_set(norm_text: str) -> set[str]:
    toks: set[str] = set()
    for w in (norm_text or "").split():
        ww = w.strip()
        if len(ww) >= 3:
            toks.add(ww)
    return toks


def _overlap_score(a: str, b: str) -> int:
    """
    Deterministic overlap between two Arabic-normalized strings.

    Score = number of shared tokens (len>=3).
    """
    sa = _token_set(a)
    sb = _token_set(b)
    if not sa or not sb:
        return 0
    return len(sa.intersection(sb))


def _clip_to_word_budget(
    *,
    raw_text: str,
    start: int,
    end: int,
    max_words: int = 25,
) -> tuple[int, int, str]:
    """Clip a raw character span to a max word budget.

    Returns:
        (new_start, new_end, quote) where quote == raw_text[new_start:new_end]

    Reason:
    - UI requires highlightable quotes.
    - Eval hard gate requires quote <= 25 words.
    - We must keep offsets consistent with the returned quote (no ellipsis).
    """
    s = max(0, int(start))
    e = max(s, int(end))
    segment = (raw_text or "")[s:e]
    if not segment.strip():
        return s, s, ""

    # Walk tokens while keeping character offsets stable.
    words = 0
    i = 0
    n = len(segment)
    in_word = False
    last_good = 0
    while i < n:
        ch = segment[i]
        if ch.isspace():
            if in_word:
                in_word = False
                last_good = i
                if words >= max_words:
                    break
            i += 1
            continue

        # non-space
        if not in_word:
            in_word = True
            words += 1
            if words > max_words:
                # stop before this word begins
                break
        i += 1

    # If we never hit whitespace boundary after last word, use current i.
    cut = last_good if (words >= max_words and last_good > 0) else i
    new_end = s + max(0, cut)
    quote = (raw_text or "")[s:new_end].strip()

    # Ensure we don't exceed budget due to punctuation/spacing oddities.
    if len([w for w in quote.split() if w]) > max_words:
        # Conservative fallback: split by whitespace to first max_words and re-find.
        keep = " ".join([w for w in quote.split() if w][:max_words]).strip()
        pos = (raw_text or "").find(keep, s, e)
        if pos >= 0:
            return pos, pos + len(keep), (raw_text or "")[pos : pos + len(keep)]
        quote = keep
        new_end = min(e, s + len(keep))
    return s, new_end, (raw_text or "")[s:new_end]


def resolve_span_from_quote(
    *,
    chunk_id: str,
    chunk_text_ar: str,
    quote: str,
) -> SpanResolution:
    """
    Resolve offsets by exact quote containment (Arabic-normalized), then fallback to raw find.

    Hard gate:
    - If we cannot locate deterministically, return unresolved with null offsets.
    """
    raw_text = chunk_text_ar or ""
    q = (quote or "").strip()
    if not raw_text or not q:
        return SpanResolution(
            chunk_id=chunk_id,
            quote=(q or ""),
            span_start=None,
            span_end=None,
            status="unresolved",
            method="missing_input",
            score=0.0,
        )

    # Fast path: raw substring search (exact)
    i = raw_text.find(q)
    if i >= 0:
        return SpanResolution(
            chunk_id=chunk_id,
            quote=q,
            span_start=i,
            span_end=i + len(q),
            status="resolved",
            method="raw_substring",
            score=float(len(q)),
        )

    # Normalized containment search (best-effort):
    # We cannot reliably map normalized offsets back to raw offsets, so we do NOT return offsets here.
    # Instead, we fall back to sentence overlap (below) in the caller if needed.
    return SpanResolution(
        chunk_id=chunk_id,
        quote=q,
        span_start=None,
        span_end=None,
        status="unresolved",
        method="quote_not_found",
        score=0.0,
    )


def resolve_span_by_sentence_overlap(
    *,
    chunk_id: str,
    chunk_text_ar: str,
    anchor_text_ar: str,
    min_overlap_tokens: int = 2,
    max_quote_words: int = 25,
) -> SpanResolution:
    """
    Choose the best sentence span by token overlap with an anchor text (usually answer_ar).

    Deterministic tie-breakers:
    - higher overlap score
    - shorter span length
    - lower span_start

    Hard gate:
    - If best overlap < min_overlap_tokens, return unresolved with null offsets.
    """
    raw = chunk_text_ar or ""
    if not raw:
        return SpanResolution(
            chunk_id=chunk_id,
            quote="",
            span_start=None,
            span_end=None,
            status="unresolved",
            method="empty_chunk",
            score=0.0,
        )

    anchor_n = normalize_for_matching(anchor_text_ar or "")
    if not anchor_n:
        # No anchor: we can provide a preview quote but cannot claim offsets are meaningful.
        preview = " ".join((raw or "").split()[:max_quote_words]).strip()
        return SpanResolution(
            chunk_id=chunk_id,
            quote=preview,
            span_start=None,
            span_end=None,
            status="unresolved",
            method="empty_anchor",
            score=0.0,
        )

    best = None
    best_score = -1
    best_len = 10**9
    best_start = 10**9

    for sp in sentence_spans(raw, max_spans=64):
        txt = span_text(raw, sp)
        txt_n = normalize_for_matching(txt)
        sc = _overlap_score(txt_n, anchor_n)
        span_len = max(0, sp.end - sp.start)
        if (sc > best_score) or (sc == best_score and span_len < best_len) or (
            sc == best_score and span_len == best_len and sp.start < best_start
        ):
            best = sp
            best_score = sc
            best_len = span_len
            best_start = sp.start

    if best is None or best_score < int(min_overlap_tokens):
        preview = " ".join((raw or "").split()[:max_quote_words]).strip()
        return SpanResolution(
            chunk_id=chunk_id,
            quote=preview,
            span_start=None,
            span_end=None,
            status="unresolved",
            method="sentence_overlap_below_threshold",
            score=float(max(best_score, 0)),
        )

    # Clip to the quote budget deterministically while keeping offsets consistent.
    s2, e2, q = _clip_to_word_budget(
        raw_text=raw,
        start=int(best.start),
        end=int(best.end),
        max_words=int(max_quote_words),
    )

    return SpanResolution(
        chunk_id=chunk_id,
        quote=q,
        span_start=int(s2) if q else None,
        span_end=int(e2) if q else None,
        status="resolved",
        method="sentence_overlap",
        score=float(best_score),
    )

