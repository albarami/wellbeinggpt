"""Debug: count cross-pillar SCHOLAR_LINK edges with justification spans."""

from __future__ import annotations

import asyncio
import sys

from sqlalchemy import text

from apps.api.core.database import get_session
from eval.datasets.source_loader import load_dotenv_if_present


async def _run() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    except Exception:
        pass
    load_dotenv_if_present()
    async with get_session() as session:
        row = (
            await session.execute(
                text(
                    """
                    SELECT
                      COUNT(*) AS total,
                      COUNT(*) FILTER (WHERE from_id <> to_id) AS cross_pillar
                    FROM edge
                    WHERE status='approved'
                      AND from_type='pillar' AND to_type='pillar'
                      AND rel_type='SCHOLAR_LINK'
                      AND relation_type IS NOT NULL
                      AND EXISTS (SELECT 1 FROM edge_justification_span s WHERE s.edge_id=edge.id)
                    """
                )
            )
        ).fetchone()
        print("total_pillar_to_pillar_semantic:", int(getattr(row, "total", 0) or 0))
        print("cross_pillar_semantic:", int(getattr(row, "cross_pillar", 0) or 0))

        rows = (
            await session.execute(
                text(
                    """
                    SELECT from_id, to_id, relation_type, COUNT(*) AS c
                    FROM edge
                    WHERE status='approved'
                      AND from_type='pillar' AND to_type='pillar'
                      AND rel_type='SCHOLAR_LINK'
                      AND relation_type IS NOT NULL
                      AND EXISTS (SELECT 1 FROM edge_justification_span s WHERE s.edge_id=edge.id)
                    GROUP BY from_id, to_id, relation_type
                    ORDER BY c DESC, from_id, to_id, relation_type
                    LIMIT 25
                    """
                )
            )
        ).fetchall()
        for r in rows:
            print(f"{r.from_id}->{r.to_id} {r.relation_type} count={int(r.c)}")


def main() -> None:
    asyncio.run(_run())


if __name__ == "__main__":
    main()

