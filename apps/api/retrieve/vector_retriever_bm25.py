"""
Local BM25 Retriever (no Azure Search, no embeddings)

Why:
- User does not have Azure AI Search and will not provision it.
- We still need a "vector-like" semantic-ish retrieval signal for:
  - `/search/vector` endpoints
  - hybrid retriever "vector-inferred anchoring"
  - retrieval quality tests (no skips)

This implementation uses BM25 over real chunk texts stored in Postgres.
It is deterministic and requires no external services.
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Any, Optional

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from apps.api.retrieve.normalize_ar import get_arabic_stopwords, normalize_for_matching


def _bm25_idf(n_docs: int, doc_freq: int) -> float:
    # Standard BM25 idf variant; add +1 inside log to avoid negative idf.
    return math.log(1.0 + ((n_docs - doc_freq + 0.5) / (doc_freq + 0.5)))


def _bm25_score(
    *,
    tf: int,
    df: int,
    doc_len: int,
    avg_doc_len: float,
    n_docs: int,
    k1: float = 1.2,
    b: float = 0.75,
) -> float:
    if tf <= 0 or doc_len <= 0 or n_docs <= 0:
        return 0.0
    idf = _bm25_idf(n_docs, df)
    denom = tf + k1 * (1.0 - b + b * (doc_len / max(avg_doc_len, 1e-9)))
    return idf * ((tf * (k1 + 1.0)) / max(denom, 1e-9))


def _tokens_for_bm25(text: str) -> list[str]:
    """
    Tokenize for BM25:
    - Arabic words (stopwords removed)
    - Digits (kept) so verse refs like "الحجرات: 6" work.
    """
    if not text:
        return []
    normalized = normalize_for_matching(text)

    # Arabic words + numbers
    import re

    tokens = re.findall(r"[\u0600-\u06FF]+|\d+", normalized)
    if not tokens:
        return []

    stop = get_arabic_stopwords()
    out: list[str] = []
    for t in tokens:
        # Keep numeric tokens
        if t.isdigit():
            out.append(t)
            continue
        if t in stop or len(t) <= 1:
            continue
        out.append(t)
    return out


async def bm25_search(
    session: AsyncSession,
    query: str,
    top_k: int = 10,
    entity_types: Optional[list[str]] = None,
    chunk_types: Optional[list[str]] = None,
) -> list[dict[str, Any]]:
    """
    Search chunks using BM25 over Arabic tokens.

    Returns evidence packets compatible with the project's contract.
    """
    q_tokens = _tokens_for_bm25(query or "")
    if not q_tokens:
        return []

    # Fetch candidate chunks from DB (text + metadata).
    where = ["c.text_ar IS NOT NULL", "c.text_ar <> ''"]
    params: dict[str, Any] = {}

    if entity_types:
        where.append("c.entity_type = ANY(:entity_types)")
        params["entity_types"] = entity_types
    if chunk_types:
        where.append("c.chunk_type = ANY(:chunk_types)")
        params["chunk_types"] = chunk_types

    sql = f"""
        SELECT c.chunk_id, c.entity_type, c.entity_id, c.chunk_type, c.text_ar, c.source_doc_id, c.source_anchor
        FROM chunk c
        WHERE {' AND '.join(where)}
    """
    rows = (await session.execute(text(sql), params)).fetchall()
    if not rows:
        return []

    # Pull refs for all candidate chunks, so ref-string queries (e.g. "آل عمران: 200")
    # work even if the reference isn't present verbatim in chunk.text_ar.
    chunk_ids_all = [str(r.chunk_id) for r in rows]
    refs_by_chunk: dict[str, list[str]] = {cid: [] for cid in chunk_ids_all}
    ref_rows = (
        await session.execute(
            text(
                """
                SELECT chunk_id, ref
                FROM chunk_ref
                WHERE chunk_id = ANY(:chunk_ids)
                """
            ),
            {"chunk_ids": chunk_ids_all},
        )
    ).fetchall()
    for rr in ref_rows:
        refs_by_chunk[str(rr.chunk_id)].append(str(rr.ref or ""))

    # Tokenize documents and compute df for query terms only (efficient).
    # Reason: corpus is small but keep it tight and deterministic.
    doc_tokens: dict[str, list[str]] = {}
    doc_tf: dict[str, Counter[str]] = {}
    doc_len: dict[str, int] = {}
    df: Counter[str] = Counter()

    q_unique = set(q_tokens)

    for r in rows:
        cid = str(r.chunk_id)
        ref_text = " ".join(refs_by_chunk.get(cid, []))
        toks = _tokens_for_bm25(f"{str(r.text_ar or '')} {ref_text}".strip())
        doc_tokens[cid] = toks
        tfc = Counter(toks)
        doc_tf[cid] = tfc
        dl = len(toks)
        doc_len[cid] = dl
        for t in q_unique:
            if tfc.get(t, 0) > 0:
                df[t] += 1

    n_docs = len(rows)
    avg_len = (sum(doc_len.values()) / max(n_docs, 1)) if n_docs else 0.0

    # Score documents.
    scored: list[tuple[float, Any]] = []
    for r in rows:
        cid = str(r.chunk_id)
        score = 0.0
        for t in q_unique:
            score += _bm25_score(
                tf=doc_tf[cid].get(t, 0),
                df=df.get(t, 0),
                doc_len=doc_len[cid],
                avg_doc_len=avg_len,
                n_docs=n_docs,
            )
        if score > 0.0:
            scored.append((score, r))

    scored.sort(key=lambda x: x[0], reverse=True)
    top = scored[: max(1, int(top_k or 10))]

    # Pre-fetch refs for returned chunks (typed).
    top_ids = [str(r.chunk_id) for _, r in top]
    refs_typed_by_chunk: dict[str, list[dict[str, str]]] = {cid: [] for cid in top_ids}
    ref_rows2 = (
        await session.execute(
            text(
                """
                SELECT chunk_id, ref_type, ref
                FROM chunk_ref
                WHERE chunk_id = ANY(:chunk_ids)
                """
            ),
            {"chunk_ids": top_ids},
        )
    ).fetchall()
    for rr in ref_rows2:
        refs_typed_by_chunk[str(rr.chunk_id)].append({"type": rr.ref_type, "ref": rr.ref})

    # Build evidence packets.
    out: list[dict[str, Any]] = []
    for score, r in top:
        out.append(
            {
                "chunk_id": str(r.chunk_id),
                "entity_type": str(r.entity_type),
                "entity_id": str(r.entity_id),
                "chunk_type": str(r.chunk_type),
                "text_ar": str(r.text_ar or ""),
                "source_doc_id": str(r.source_doc_id),
                "source_anchor": str(r.source_anchor or ""),
                "refs": refs_typed_by_chunk.get(str(r.chunk_id), []),
                "score": float(score),
                "backend": "bm25",
            }
        )
    return out

