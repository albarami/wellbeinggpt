"""Chunk sentence span persistence.

Stores deterministic sentence spans for each chunk so citations can reference
stable offsets without doing fuzzy "best span" discovery.

This is invoked during ingestion/bootstrap (best-effort).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from apps.api.ingest.sentence_spans import sentence_spans


@dataclass(frozen=True)
class StoredSpan:
    chunk_id: str
    span_index: int
    span_start: int
    span_end: int
    text_hash: str


def _hash_text(s: str) -> str:
    return hashlib.sha256((s or "").encode("utf-8")).hexdigest()


async def ensure_chunk_span_table(session: AsyncSession) -> None:
    """Create chunk_span table if missing (idempotent)."""
    await session.execute(
        text(
            """
            CREATE TABLE IF NOT EXISTS chunk_span (
              id uuid PRIMARY KEY DEFAULT uuid_generate_v4(),
              chunk_id VARCHAR(50) NOT NULL,
              span_index INTEGER NOT NULL,
              span_start INTEGER NOT NULL,
              span_end INTEGER NOT NULL,
              text_hash VARCHAR(64) NOT NULL,
              created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
              UNIQUE(chunk_id, span_index)
            );
            """
        )
    )
    await session.execute(text("CREATE INDEX IF NOT EXISTS idx_chunk_span_chunk_id ON chunk_span(chunk_id);"))


async def populate_chunk_spans_for_source(session: AsyncSession, source_doc_id: str) -> int:
    """Populate spans for all chunks in a source_doc_id.

    Behavior:
    - Deletes existing spans for those chunks (idempotent)
    - Inserts deterministic sentence spans

    Returns:
        Number of inserted span rows.
    """
    await ensure_chunk_span_table(session)

    # Delete existing spans for chunks in this source.
    await session.execute(
        text(
            """
            DELETE FROM chunk_span
            WHERE chunk_id IN (SELECT chunk_id FROM chunk WHERE source_doc_id=:sd)
            """
        ),
        {"sd": source_doc_id},
    )

    rows = (
        await session.execute(
            text(
                """
                SELECT chunk_id, text_ar
                FROM chunk
                WHERE source_doc_id=:sd
                  AND text_ar IS NOT NULL AND text_ar <> ''
                ORDER BY chunk_id
                """
            ),
            {"sd": source_doc_id},
        )
    ).fetchall()

    inserted = 0
    for r in rows:
        cid = str(r.chunk_id)
        txt = str(r.text_ar or "")
        spans = sentence_spans(txt, max_spans=64)
        for idx, sp in enumerate(spans):
            sub = txt[sp.start : sp.end]
            await session.execute(
                text(
                    """
                    INSERT INTO chunk_span (chunk_id, span_index, span_start, span_end, text_hash)
                    VALUES (:cid, :i, :s, :e, :h)
                    ON CONFLICT (chunk_id, span_index) DO UPDATE SET
                      span_start=EXCLUDED.span_start,
                      span_end=EXCLUDED.span_end,
                      text_hash=EXCLUDED.text_hash
                    """
                ),
                {
                    "cid": cid,
                    "i": int(idx),
                    "s": int(sp.start),
                    "e": int(sp.end),
                    "h": _hash_text(sub),
                },
            )
            inserted += 1

    return inserted
