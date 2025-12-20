"""Quick DB probe for framework chunks containing given substrings.

Run examples:
  python -m scripts.query_framework_chunks --source-name framework_2025-10_v1 --p1 الجسد --p2 الروح
"""

from __future__ import annotations

import argparse
import asyncio

from sqlalchemy import text

from apps.api.core.database import get_session
from eval.datasets.source_loader import load_dotenv_if_present


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


async def _run(*, source_name: str, p1: str, p2: str, limit: int) -> None:
    load_dotenv_if_present()
    async with get_session() as session:
        sd = await _source_doc_id_for_pattern(session=session, source_name=source_name)
        if not sd:
            raise SystemExit(f"Could not find source_document matching file_name ILIKE '%{source_name}%'")

        res = await session.execute(
            text(
                """
                SELECT chunk_id, chunk_type, left(text_ar, 500) AS preview
                FROM chunk
                WHERE source_doc_id::text = :sd
                  AND text_ar ILIKE :p1
                  AND text_ar ILIKE :p2
                ORDER BY chunk_id
                LIMIT :limit
                """
            ),
            {"sd": sd, "p1": f"%{p1}%", "p2": f"%{p2}%", "limit": int(limit)},
        )
        rows = res.fetchall()
        print(f"matches={len(rows)}")
        for r in rows:
            print(f"- {r.chunk_id} [{r.chunk_type}]")
            print(str(r.preview or "").replace("\n", " ")[:500])
            print("")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-name", default="framework_2025-10_v1")
    ap.add_argument("--p1", required=True)
    ap.add_argument("--p2", required=True)
    ap.add_argument("--limit", type=int, default=5)
    args = ap.parse_args()
    asyncio.run(_run(source_name=str(args.source_name), p1=str(args.p1), p2=str(args.p2), limit=int(args.limit)))


if __name__ == "__main__":
    main()

