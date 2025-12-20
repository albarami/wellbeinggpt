"""
Reference coverage utilities.

Turns a ref node (e.g., quran:البقرة:255) into a coverage map:
- which entities mention it
- which pillars it spans
- counts for UI ranking and drill-down
"""

from __future__ import annotations

from typing import Any

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession


async def ref_coverage(session: AsyncSession, ref_node_id: str, limit: int = 200) -> dict[str, Any]:
    """
    Compute coverage for a ref node.

    Args:
        session: DB session.
        ref_node_id: Ref node id '<type>:<ref_norm>'.
        limit: Limit entities returned.

    Returns:
        dict with entities + pillar counts. Names are best-effort.
    """
    rows = (
        await session.execute(
            text(
                """
                SELECT e.from_type AS entity_type, e.from_id AS entity_id
                FROM edge e
                WHERE e.rel_type = 'MENTIONS_REF'
                  AND e.to_type = 'ref'
                  AND e.to_id = :rid
                  AND e.status = 'approved'
                ORDER BY e.from_type, e.from_id
                LIMIT :lim
                """
            ),
            {"rid": ref_node_id, "lim": limit},
        )
    ).fetchall()

    entities = [{"entity_type": r.entity_type, "entity_id": r.entity_id} for r in rows]

    # Best-effort names + pillar mapping (only for core_value/sub_value).
    # Keep queries simple and bounded.
    pillar_rows = (
        await session.execute(
            text(
                """
                WITH ents AS (
                    SELECT e.from_type AS entity_type, e.from_id AS entity_id
                    FROM edge e
                    WHERE e.rel_type='MENTIONS_REF'
                      AND e.to_type='ref'
                      AND e.to_id=:rid
                      AND e.status='approved'
                    LIMIT :lim
                )
                SELECT
                    p.id::text AS pillar_id,
                    p.name_ar AS pillar_name_ar,
                    COUNT(*)::int AS entity_count
                FROM ents
                JOIN core_value cv
                  ON ents.entity_type='core_value' AND cv.id::text = ents.entity_id
                JOIN pillar p ON p.id=cv.pillar_id
                GROUP BY p.id, p.name_ar
                UNION ALL
                SELECT
                    p.id::text AS pillar_id,
                    p.name_ar AS pillar_name_ar,
                    COUNT(*)::int AS entity_count
                FROM ents
                JOIN sub_value sv
                  ON ents.entity_type='sub_value' AND sv.id::text = ents.entity_id
                JOIN core_value cv ON cv.id=sv.core_value_id
                JOIN pillar p ON p.id=cv.pillar_id
                GROUP BY p.id, p.name_ar
                """
            ),
            {"rid": ref_node_id, "lim": limit},
        )
    ).fetchall()

    pillar_counts: dict[str, dict[str, Any]] = {}
    for r in pillar_rows:
        pid = str(r.pillar_id)
        if pid not in pillar_counts:
            pillar_counts[pid] = {"pillar_id": pid, "pillar_name_ar": r.pillar_name_ar, "entity_count": 0}
        pillar_counts[pid]["entity_count"] += int(r.entity_count)

    return {
        "ref_node_id": ref_node_id,
        "entities": entities,
        "pillars": list(pillar_counts.values()),
    }






