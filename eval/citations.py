"""Citation span resolution.

Requirement: citations must point to real chunk IDs and valid (start,end) offsets.

We do NOT "search for best spans". Instead we choose from deterministic
sentence spans derived from the stored chunk text.

In the long term, these spans are persisted at ingestion time (chunk_span table).
This resolver is used by the eval runner and scorers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from apps.api.retrieve.normalize_ar import extract_arabic_words, normalize_for_matching
from apps.api.ingest.sentence_spans import sentence_spans, span_text
from eval.types import EvalCitation


@dataclass(frozen=True)
class ChunkText:
    chunk_id: str
    text_ar: str


async def fetch_chunk_text(session: AsyncSession, chunk_id: str) -> Optional[ChunkText]:
    row = (
        await session.execute(
            text(
                """
                SELECT chunk_id, text_ar
                FROM chunk
                WHERE chunk_id = :cid
                """
            ),
            {"cid": chunk_id},
        )
    ).fetchone()
    if not row:
        return None
    return ChunkText(chunk_id=str(row.chunk_id), text_ar=str(row.text_ar or ""))


def quote_25_words(text: str) -> str:
    words = [w for w in (text or "").split() if w]
    if len(words) <= 25:
        return " ".join(words)
    return " ".join(words[:25])


def _overlap_score(query: str, candidate: str) -> int:
    q_terms = [t for t in extract_arabic_words(query or "") if len(t) >= 3]
    if not q_terms:
        return 0
    cand_norm = normalize_for_matching(candidate or "")
    if not cand_norm:
        return 0
    score = 0
    for t in q_terms[:20]:
        if normalize_for_matching(t) in cand_norm:
            score += 1
    return score


async def citation_for_chunk_first_sentence(
    session: AsyncSession, chunk_id: str
) -> Optional[EvalCitation]:
    ct = await fetch_chunk_text(session, chunk_id)
    if not ct or not (ct.text_ar or "").strip():
        return None
    # Prefer persisted spans when available.
    row = None
    try:
        row = (
            await session.execute(
                text(
                    """
                    SELECT span_start, span_end
                    FROM chunk_span
                    WHERE chunk_id = :cid AND span_index = 0
                    LIMIT 1
                    """
                ),
                {"cid": chunk_id},
            )
        ).fetchone()
    except Exception:
        row = None

    if row:
        sp_start = int(row.span_start)
        sp_end = int(row.span_end)
        sub = ct.text_ar[sp_start:sp_end]
        q = quote_25_words(sub)
        return EvalCitation(source_id=chunk_id, span_start=sp_start, span_end=sp_end, quote=q)

    spans = sentence_spans(ct.text_ar)
    if not spans:
        return None
    sp = spans[0]
    q = quote_25_words(span_text(ct.text_ar, sp))
    return EvalCitation(source_id=chunk_id, span_start=sp.start, span_end=sp.end, quote=q)


async def citation_for_chunk_best_sentence(
    session: AsyncSession, *, chunk_id: str, query_text: str
) -> Optional[EvalCitation]:
    """
    Select a citation span deterministically from precomputed sentence spans.

    This is NOT fuzzy search: we only choose among persisted sentence offsets and
    score by exact normalized term overlap.
    """
    ct = await fetch_chunk_text(session, chunk_id)
    if not ct or not (ct.text_ar or "").strip():
        return None

    # Prefer persisted spans list.
    try:
        rows = (
            await session.execute(
                text(
                    """
                    SELECT span_index, span_start, span_end
                    FROM chunk_span
                    WHERE chunk_id = :cid
                    ORDER BY span_index
                    """
                ),
                {"cid": chunk_id},
            )
        ).fetchall()
    except Exception:
        rows = []

    best = None
    best_score = -1

    if rows:
        for r in rows:
            s = int(r.span_start)
            e = int(r.span_end)
            if s < 0 or e <= s or e > len(ct.text_ar):
                continue
            sub = ct.text_ar[s:e]
            sc = _overlap_score(query_text, sub)
            if sc > best_score:
                best_score = sc
                best = (s, e, sub)
        if best is None:
            return await citation_for_chunk_first_sentence(session, chunk_id)
        s, e, sub = best
        return EvalCitation(
            source_id=chunk_id,
            span_start=s,
            span_end=e,
            quote=quote_25_words(sub),
        )

    # Fallback: recompute spans from text.
    spans = sentence_spans(ct.text_ar)
    if not spans:
        return None
    best_sp = spans[0]
    best_score = _overlap_score(query_text, span_text(ct.text_ar, best_sp))
    for sp in spans[1:]:
        sub = span_text(ct.text_ar, sp)
        sc = _overlap_score(query_text, sub)
        if sc > best_score:
            best_score = sc
            best_sp = sp
    sub = span_text(ct.text_ar, best_sp)
    return EvalCitation(
        source_id=chunk_id,
        span_start=best_sp.start,
        span_end=best_sp.end,
        quote=quote_25_words(sub),
    )
