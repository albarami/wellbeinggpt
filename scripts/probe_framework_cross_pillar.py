"""Probe framework chunks for cross-pillar cue phrases (no CLI args).

Reason: Windows PowerShell quoting can be fragile with Arabic CLI args.
This script prints counts and a few sample chunk previews for key phrases.
"""

from __future__ import annotations

import asyncio
import sys

from sqlalchemy import text

from apps.api.core.database import get_session
from eval.datasets.source_loader import load_dotenv_if_present


PHRASE_PAIRS: list[tuple[str, str]] = [
    ("الجسد", "الروح"),
    ("لا يفصل", "الروح"),
    ("البدنية", "الروحية"),
    ("وسيلة", "الروحية"),
    ("وسيلة", "العاطفية"),
    ("وسيلة", "الفكرية"),
    ("وسيلة", "الاجتماعية"),
]


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


async def _probe(*, source_name: str = "framework_2025-10_v1") -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    except Exception:
        pass
    load_dotenv_if_present()
    async with get_session() as session:
        sd = await _source_doc_id_for_pattern(session=session, source_name=source_name)
        if not sd:
            raise SystemExit(f"Could not find source_document matching file_name ILIKE '%{source_name}%'")

        for p1, p2 in PHRASE_PAIRS:
            res = await session.execute(
                text(
                    """
                    SELECT chunk_id, chunk_type, left(text_ar, 350) AS preview
                    FROM chunk
                    WHERE source_doc_id::text = :sd
                      AND text_ar ILIKE :p1
                      AND text_ar ILIKE :p2
                    ORDER BY chunk_id
                    LIMIT 3
                    """
                ),
                {"sd": sd, "p1": f"%{p1}%", "p2": f"%{p2}%"},
            )
            rows = res.fetchall()
            # Count total
            c = (
                await session.execute(
                    text(
                        """
                        SELECT COUNT(*) AS c
                        FROM chunk
                        WHERE source_doc_id::text = :sd
                          AND text_ar ILIKE :p1
                          AND text_ar ILIKE :p2
                        """
                    ),
                    {"sd": sd, "p1": f"%{p1}%", "p2": f"%{p2}%"},
                )
            ).fetchone()
            total = int(getattr(c, "c", 0) or 0)
            print(f"\nPAIR '{p1}' + '{p2}': matches={total}")
            for r in rows:
                preview = str(r.preview or "").replace("\n", " ")
                print(f"- {r.chunk_id} [{r.chunk_type}] {preview[:350]}")


def main() -> None:
    asyncio.run(_probe())


if __name__ == "__main__":
    main()

