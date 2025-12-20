"""Mine grounded cross-pillar semantic edges from the framework chunks (framework-only).

This script scans the ingested framework chunks and inserts:
- edge rows: rel_type='SCHOLAR_LINK' with a semantic relation_type (e.g. ENABLES)
- edge_justification_span rows with the exact sentence offsets and quote

Hard gate:
- No span -> no edge.

Run:
  python -m scripts.mine_framework_semantic_edges --source-name framework_2025-10_v1
"""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import dataclass
from typing import Any

from sqlalchemy import text

from apps.api.core.database import get_session
from apps.api.graph.framework_edge_miner import extract_semantic_edges_from_chunk, upsert_mined_edges
from eval.datasets.source_loader import load_dotenv_if_present


@dataclass(frozen=True)
class MineResult:
    chunks_scanned: int
    candidate_edges: int
    inserted_edges: int
    inserted_spans: int


async def _source_doc_id_for_pattern(*, session, source_name: str) -> str | None:
    row = (
        await session.execute(
            text(
                """
                SELECT id::text AS id
                FROM source_document
                WHERE file_name ILIKE :p
                ORDER BY created_at DESC
                LIMIT 1
                """
            ),
            {"p": f"%{source_name}%"},
        )
    ).fetchone()
    return str(row.id) if row and getattr(row, "id", None) else None


async def _iter_framework_chunks(*, session, source_doc_id: str, limit: int | None) -> list[dict[str, Any]]:
    q = """
        SELECT chunk_id, chunk_type, text_ar
        FROM chunk
        WHERE source_doc_id::text = :sd
        ORDER BY chunk_id
    """
    params: dict[str, Any] = {"sd": str(source_doc_id)}
    if limit is not None and int(limit) > 0:
        q += " LIMIT :limit"
        params["limit"] = int(limit)
    res = await session.execute(text(q), params)
    return [dict(r._mapping) for r in res.fetchall()]


async def mine_and_insert(
    *,
    source_name: str,
    dry_run: bool,
    limit_chunks: int | None,
) -> MineResult:
    load_dotenv_if_present()

    async with get_session() as session:
        sd = await _source_doc_id_for_pattern(session=session, source_name=source_name)
        if not sd:
            raise SystemExit(f"Could not find source_document matching file_name ILIKE '%{source_name}%'")

        chunks = await _iter_framework_chunks(session=session, source_doc_id=sd, limit=limit_chunks)
        mined_all = []
        for c in chunks:
            txt = str(c.get("text_ar") or "")
            if not txt.strip():
                continue
            mined_all.extend(extract_semantic_edges_from_chunk(chunk_id=str(c.get("chunk_id") or ""), text_ar=txt))

        mined_all = [e for e in mined_all if e.from_pillar_id != e.to_pillar_id]
        if dry_run:
            return MineResult(
                chunks_scanned=len(chunks),
                candidate_edges=len(mined_all),
                inserted_edges=0,
                inserted_spans=0,
            )

        summary = await upsert_mined_edges(
            session=session,
            mined=mined_all,
            created_by="framework_semantic_edge_miner",
            strength_score=0.8,
        )
        await session.commit()

        return MineResult(
            chunks_scanned=len(chunks),
            candidate_edges=len(mined_all),
            inserted_edges=int(summary.get("inserted_edges") or 0),
            inserted_spans=int(summary.get("inserted_edge_spans") or 0),
        )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-name", default="framework_2025-10_v1", help="Substring match for source_document.file_name")
    ap.add_argument("--dry-run", action="store_true", help="Do not write to DB")
    ap.add_argument("--limit-chunks", type=int, default=0, help="Optional limit for scanning chunks (0 = no limit)")
    args = ap.parse_args()

    res = asyncio.run(
        mine_and_insert(
            source_name=str(args.source_name),
            dry_run=bool(args.dry_run),
            limit_chunks=(int(args.limit_chunks) if int(args.limit_chunks) > 0 else None),
        )
    )
    print(
        f"chunks_scanned={res.chunks_scanned} candidate_edges={res.candidate_edges} "
        f"inserted_edges={res.inserted_edges} inserted_spans={res.inserted_spans}"
    )


if __name__ == "__main__":
    main()

