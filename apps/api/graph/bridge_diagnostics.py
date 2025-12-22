"""Diagnostics for grounded cross-pillar semantic edges.

Reason:
- Stakeholder cross-pillar answers require *grounded* semantic edges (SCHOLAR_LINK + edge_justification_span).
- When `used_edges = []`, we need to distinguish:
  (A) No grounded cross-pillar edges exist in DB (knowledge gap), vs
  (B) Edges exist but selection/filtering removes them (tuning issue).
"""

from __future__ import annotations

from typing import Any

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession


async def count_value_level_scholar_links(*, session: AsyncSession) -> dict[str, int]:
    """Count value-level SCHOLAR_LINK edges (core_value/sub_value endpoints).

    Returns:
        Dict with total and grounded counts, plus cross-pillar breakdown.
    """
    result = await session.execute(
        text(
            """
            WITH value_edges AS (
              SELECT
                e.id,
                e.from_type,
                e.from_id,
                e.to_type,
                e.to_id,
                -- Resolve pillar for from endpoint
                CASE
                  WHEN e.from_type = 'pillar' THEN e.from_id
                  WHEN e.from_type = 'core_value' THEN cv_from.pillar_id
                  WHEN e.from_type = 'sub_value' THEN cv_from2.pillar_id
                  ELSE NULL
                END AS from_pillar_id,
                -- Resolve pillar for to endpoint
                CASE
                  WHEN e.to_type = 'pillar' THEN e.to_id
                  WHEN e.to_type = 'core_value' THEN cv_to.pillar_id
                  WHEN e.to_type = 'sub_value' THEN cv_to2.pillar_id
                  ELSE NULL
                END AS to_pillar_id
              FROM edge e
              LEFT JOIN core_value cv_from ON (e.from_type='core_value' AND cv_from.id = e.from_id)
              LEFT JOIN sub_value sv_from ON (e.from_type='sub_value' AND sv_from.id = e.from_id)
              LEFT JOIN core_value cv_from2 ON (sv_from.core_value_id = cv_from2.id)

              LEFT JOIN core_value cv_to ON (e.to_type='core_value' AND cv_to.id = e.to_id)
              LEFT JOIN sub_value sv_to ON (e.to_type='sub_value' AND sv_to.id = e.to_id)
              LEFT JOIN core_value cv_to2 ON (sv_to.core_value_id = cv_to2.id)
              WHERE e.rel_type = 'SCHOLAR_LINK'
                AND e.relation_type IS NOT NULL
                AND (e.from_type IN ('core_value', 'sub_value')
                     OR e.to_type IN ('core_value', 'sub_value'))
            )
            SELECT
              COUNT(*) AS total_value_edges,
              COUNT(*) FILTER (WHERE EXISTS (
                SELECT 1 FROM edge_justification_span js WHERE js.edge_id = ve.id
              )) AS grounded_value_edges,
              COUNT(*) FILTER (WHERE ve.from_pillar_id IS NOT NULL
                AND ve.to_pillar_id IS NOT NULL
                AND ve.from_pillar_id <> ve.to_pillar_id) AS cross_pillar_value_edges,
              COUNT(*) FILTER (WHERE ve.from_pillar_id IS NOT NULL
                AND ve.to_pillar_id IS NOT NULL
                AND ve.from_pillar_id <> ve.to_pillar_id
                AND EXISTS (SELECT 1 FROM edge_justification_span js WHERE js.edge_id = ve.id)
              ) AS grounded_cross_pillar_value_edges
            FROM value_edges ve
            """
        )
    )
    row = result.fetchone()
    return {
        "total_value_edges": int(getattr(row, "total_value_edges", 0) or 0),
        "grounded_value_edges": int(getattr(row, "grounded_value_edges", 0) or 0),
        "cross_pillar_value_edges": int(getattr(row, "cross_pillar_value_edges", 0) or 0),
        "grounded_cross_pillar_value_edges": int(getattr(row, "grounded_cross_pillar_value_edges", 0) or 0),
    }


async def count_grounded_cross_pillar_scholar_links(*, session: AsyncSession) -> int:
    """Count grounded SCHOLAR_LINK edges that connect different pillars.

    A "cross-pillar" edge is defined as:
    - rel_type = 'SCHOLAR_LINK'
    - relation_type IS NOT NULL
    - has >=1 edge_justification_span
    - resolves to a pillar on both endpoints, and those pillar_ids differ

    Args:
        session: DB session.

    Returns:
        Count of grounded cross-pillar semantic edges.
    """

    row = (
        await session.execute(
            text(
                """
                WITH edge_with_pillars AS (
                  SELECT
                    e.id,
                    e.rel_type,
                    e.relation_type,
                    -- resolve pillar for from endpoint
                    CASE
                      WHEN e.from_type = 'pillar' THEN e.from_id
                      WHEN e.from_type = 'core_value' THEN cv_from.pillar_id
                      WHEN e.from_type = 'sub_value' THEN cv_from2.pillar_id
                      ELSE NULL
                    END AS from_pillar_id,
                    -- resolve pillar for to endpoint
                    CASE
                      WHEN e.to_type = 'pillar' THEN e.to_id
                      WHEN e.to_type = 'core_value' THEN cv_to.pillar_id
                      WHEN e.to_type = 'sub_value' THEN cv_to2.pillar_id
                      ELSE NULL
                    END AS to_pillar_id
                  FROM edge e
                  LEFT JOIN core_value cv_from ON (e.from_type='core_value' AND cv_from.id = e.from_id)
                  LEFT JOIN sub_value sv_from ON (e.from_type='sub_value' AND sv_from.id = e.from_id)
                  LEFT JOIN core_value cv_from2 ON (sv_from.core_value_id = cv_from2.id)

                  LEFT JOIN core_value cv_to ON (e.to_type='core_value' AND cv_to.id = e.to_id)
                  LEFT JOIN sub_value sv_to ON (e.to_type='sub_value' AND sv_to.id = e.to_id)
                  LEFT JOIN core_value cv_to2 ON (sv_to.core_value_id = cv_to2.id)
                  WHERE e.rel_type = 'SCHOLAR_LINK' AND e.relation_type IS NOT NULL
                )
                SELECT COUNT(DISTINCT ewp.id) AS c
                FROM edge_with_pillars ewp
                WHERE ewp.from_pillar_id IS NOT NULL
                  AND ewp.to_pillar_id IS NOT NULL
                  AND ewp.from_pillar_id <> ewp.to_pillar_id
                  AND EXISTS (
                    SELECT 1 FROM edge_justification_span js WHERE js.edge_id = ewp.id
                  )
                """
            )
        )
    ).fetchone()
    return int(getattr(row, "c", 0) or 0)


async def sample_grounded_cross_pillar_scholar_links(
    *, session: AsyncSession, limit: int = 20
) -> list[dict[str, Any]]:
    """Return a small sample of grounded cross-pillar scholar links for debugging."""

    res = await session.execute(
        text(
            """
            WITH edge_with_pillars AS (
              SELECT
                e.id::text AS edge_id,
                e.from_type, e.from_id,
                e.to_type, e.to_id,
                e.rel_type, e.relation_type,
                CASE
                  WHEN e.from_type = 'pillar' THEN e.from_id
                  WHEN e.from_type = 'core_value' THEN cv_from.pillar_id
                  WHEN e.from_type = 'sub_value' THEN cv_from2.pillar_id
                  ELSE NULL
                END AS from_pillar_id,
                CASE
                  WHEN e.to_type = 'pillar' THEN e.to_id
                  WHEN e.to_type = 'core_value' THEN cv_to.pillar_id
                  WHEN e.to_type = 'sub_value' THEN cv_to2.pillar_id
                  ELSE NULL
                END AS to_pillar_id
              FROM edge e
              LEFT JOIN core_value cv_from ON (e.from_type='core_value' AND cv_from.id = e.from_id)
              LEFT JOIN sub_value sv_from ON (e.from_type='sub_value' AND sv_from.id = e.from_id)
              LEFT JOIN core_value cv_from2 ON (sv_from.core_value_id = cv_from2.id)

              LEFT JOIN core_value cv_to ON (e.to_type='core_value' AND cv_to.id = e.to_id)
              LEFT JOIN sub_value sv_to ON (e.to_type='sub_value' AND sv_to.id = e.to_id)
              LEFT JOIN core_value cv_to2 ON (sv_to.core_value_id = cv_to2.id)
              WHERE e.rel_type = 'SCHOLAR_LINK' AND e.relation_type IS NOT NULL
            )
            SELECT
              ewp.edge_id, ewp.relation_type,
              ewp.from_pillar_id, ewp.to_pillar_id,
              (
                SELECT COUNT(*) FROM edge_justification_span js
                WHERE js.edge_id::text = ewp.edge_id
              ) AS justification_spans
            FROM edge_with_pillars ewp
            WHERE ewp.from_pillar_id IS NOT NULL
              AND ewp.to_pillar_id IS NOT NULL
              AND ewp.from_pillar_id <> ewp.to_pillar_id
              AND EXISTS (
                SELECT 1 FROM edge_justification_span js WHERE js.edge_id::text = ewp.edge_id
              )
            ORDER BY justification_spans DESC, ewp.edge_id
            LIMIT :limit
            """
        ),
        {"limit": int(limit)},
    )
    return [dict(r._mapping) for r in res.fetchall()]

