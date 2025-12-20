"""
Build Local Embeddings (no Azure, no external services)

Creates deterministic hashed TF-IDF vectors for every chunk and stores them in Postgres `embedding`.
This satisfies "vector coverage" requirements without Azure Search or an embeddings deployment.

Usage:
  python scripts/build_local_embeddings.py

Env vars:
  DATABASE_URL (required)
  LOCAL_EMBEDDING_DIMS (optional, default: 512)
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import os
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from sqlalchemy import text

from apps.api.core.database import get_session
from apps.api.retrieve.normalize_ar import get_arabic_stopwords, normalize_for_matching
from apps.api.retrieve.vector_retriever import store_embedding


def _require_env(name: str) -> str:
    v = (os.getenv(name) or "").strip()
    if not v:
        raise RuntimeError(f"Missing required env var: {name}")
    return v


def _dims() -> int:
    try:
        return int(os.getenv("LOCAL_EMBEDDING_DIMS", "512"))
    except Exception:
        return 512


def _tokenize(text: str) -> list[str]:
    if not text:
        return []
    import re

    normalized = normalize_for_matching(text)
    tokens = re.findall(r"[\u0600-\u06FF]+|\d+", normalized)
    if not tokens:
        return []
    stop = get_arabic_stopwords()
    out: list[str] = []
    for t in tokens:
        if t.isdigit():
            out.append(t)
            continue
        if t in stop or len(t) <= 1:
            continue
        out.append(t)
    return out


def _hash_bin(token: str, dims: int) -> int:
    # Stable across processes/platforms.
    h = hashlib.blake2b(token.encode("utf-8"), digest_size=4).digest()
    return int.from_bytes(h, "little") % dims


def _l2_normalize(vec: list[float]) -> list[float]:
    s = math.sqrt(sum(v * v for v in vec))
    if s <= 0.0:
        return vec
    return [v / s for v in vec]


async def build_local_embeddings() -> dict[str, Any]:
    dims = _dims()
    report: dict[str, Any] = {
        "generated_at": datetime.utcnow().isoformat(),
        "dims": dims,
        "chunks_total": 0,
        "embedded": 0,
        "model": "local_hash_tfidf",
        "status": "pending",
    }

    async with get_session() as session:
        chunk_rows = (
            await session.execute(
                text(
                    """
                    SELECT chunk_id, text_ar
                    FROM chunk
                    ORDER BY chunk_id
                    """
                )
            )
        ).fetchall()
        report["chunks_total"] = len(chunk_rows)
        if not chunk_rows:
            report["status"] = "error"
            report["error"] = "No chunks found"
            return report

        chunk_ids = [str(r.chunk_id) for r in chunk_rows]
        refs_rows = (
            await session.execute(
                text(
                    """
                    SELECT chunk_id, ref
                    FROM chunk_ref
                    WHERE chunk_id = ANY(:chunk_ids)
                    """
                ),
                {"chunk_ids": chunk_ids},
            )
        ).fetchall()
        refs_by_chunk: dict[str, list[str]] = {cid: [] for cid in chunk_ids}
        for rr in refs_rows:
            refs_by_chunk[str(rr.chunk_id)].append(str(rr.ref or ""))

        # First pass: tokenize and compute DF per bin
        doc_bins: dict[str, Counter[int]] = {}
        df = [0] * dims
        for r in chunk_rows:
            cid = str(r.chunk_id)
            ref_text = " ".join(refs_by_chunk.get(cid, []))
            toks = _tokenize(f"{str(r.text_ar or '')} {ref_text}".strip())
            counts: Counter[int] = Counter()
            for t in toks:
                counts[_hash_bin(t, dims)] += 1
            doc_bins[cid] = counts
            for b in set(counts.keys()):
                df[b] += 1

        n_docs = len(chunk_rows)
        idf = [math.log((n_docs + 1) / (d + 1)) + 1.0 for d in df]

        # Second pass: build vectors and store
        for cid, counts in doc_bins.items():
            vec = [0.0] * dims
            doc_len = sum(counts.values()) or 1
            for b, tf in counts.items():
                vec[b] = (tf / doc_len) * idf[b]
            vec = _l2_normalize(vec)
            await store_embedding(
                session=session,
                chunk_id=cid,
                vector=vec,
                model="local_hash_tfidf",
                dims=dims,
            )
            report["embedded"] += 1

        await session.commit()

    report["status"] = "complete" if report["embedded"] == report["chunks_total"] else "incomplete"
    return report


def main() -> None:
    load_dotenv()
    _require_env("DATABASE_URL")
    rep = asyncio.run(build_local_embeddings())
    out = Path("local_embedding_report.json")
    out.write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote: {out}")
    print(f"Status: {rep['status']}, embedded {rep['embedded']}/{rep['chunks_total']}, dims={rep['dims']}")


if __name__ == "__main__":
    main()





