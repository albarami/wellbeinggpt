"""
Graph analytics utilities (Postgres edges table).

These functions power "enterprise-grade" graph features:
- Central Quran/Hadith refs (by how many values they support)
- Cross-pillar discovery via ref nodes
- Lightweight concept networks for visualization
"""

from __future__ import annotations

from typing import Any, Optional

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession


async def top_ref_nodes(
    session: AsyncSession,
    evidence_type: Optional[str] = None,
    created_by: Optional[str] = None,
    top_k: int = 10,
) -> list[dict[str, Any]]:
    """
    Rank ref nodes by how many distinct entities mention them.

    Args:
        session: DB session.
        evidence_type: Optional filter: quran | hadith | book.
        top_k: Max rows.

    Returns:
        List of {ref_node_id, entity_count, evidence_count}.
    """
    params: dict[str, Any] = {"top_k": top_k}
    where = ""
    if evidence_type:
        where = " AND split_part(e.to_id, ':', 1) = :etype"
        params["etype"] = evidence_type
    if created_by:
        where += " AND e.created_by = :created_by"
        params["created_by"] = created_by

    rows = (
        await session.execute(
            text(
                f"""
                WITH ref_edges AS (
                    SELECT to_id AS ref_node_id, from_type, from_id
                    FROM edge e
                    WHERE e.rel_type = 'MENTIONS_REF'
                      AND e.to_type = 'ref'
                      AND e.status = 'approved'
                      {where}
                ),
                ev_edges AS (
                    SELECT to_id AS ref_node_id, from_id AS evidence_id
                    FROM edge e
                    WHERE e.rel_type = 'REFERS_TO'
                      AND e.to_type = 'ref'
                      AND e.status = 'approved'
                      {where}
                )
                SELECT
                    r.ref_node_id,
                    COUNT(DISTINCT (r.from_type || ':' || r.from_id)) AS entity_count,
                    COUNT(DISTINCT ev.evidence_id) AS evidence_count
                FROM ref_edges r
                LEFT JOIN ev_edges ev ON ev.ref_node_id = r.ref_node_id
                GROUP BY r.ref_node_id
                ORDER BY entity_count DESC, evidence_count DESC, r.ref_node_id
                LIMIT :top_k
                """
            ),
            params,
        )
    ).fetchall()

    return [dict(r._mapping) for r in rows]


async def concept_network(
    session: AsyncSession,
    name_pattern: str,
    depth: int = 2,
    limit_entities: int = 20,
) -> dict[str, Any]:
    """
    Build a small network centered on entities whose name matches the pattern.

    Returns nodes + edges suitable for UI visualization.
    """
    # 1) Seed entities by name match
    seeds = (
        await session.execute(
            text(
                """
                SELECT 'pillar' AS entity_type, id::text AS id, name_ar
                FROM pillar
                WHERE name_ar ILIKE :p
                UNION ALL
                SELECT 'core_value' AS entity_type, id::text AS id, name_ar
                FROM core_value
                WHERE name_ar ILIKE :p
                UNION ALL
                SELECT 'sub_value' AS entity_type, id::text AS id, name_ar
                FROM sub_value
                WHERE name_ar ILIKE :p
                LIMIT :lim
                """
            ),
            {"p": f"%{name_pattern}%", "lim": limit_entities},
        )
    ).fetchall()

    seed_nodes = [dict(r._mapping) for r in seeds]
    if not seed_nodes:
        return {"nodes": [], "edges": [], "seeds": []}

    # 2) Collect neighborhood by following approved edges up to depth
    # We do BFS in SQL by iterating levels (small limits; safe for MVP).
    nodes: dict[tuple[str, str], dict[str, Any]] = {}
    edges: list[dict[str, Any]] = []

    frontier = [(n["entity_type"], n["id"]) for n in seed_nodes]
    for n in seed_nodes:
        nodes[(n["entity_type"], n["id"])] = n

    for _ in range(depth):
        if not frontier:
            break

        # Expand from current frontier (both directions)
        rows = (
            await session.execute(
                text(
                    """
                    SELECT
                        e.from_type, e.from_id, e.rel_type, e.to_type, e.to_id
                    FROM edge e
                    WHERE e.status = 'approved'
                      AND (
                        (e.from_type, e.from_id) = ANY(:pairs)
                        OR
                        (e.to_type, e.to_id) = ANY(:pairs)
                      )
                    """
                ),
                {"pairs": frontier},
            )
        ).fetchall()

        next_frontier: list[tuple[str, str]] = []
        for r in rows:
            fr = (str(r.from_type), str(r.from_id))
            to = (str(r.to_type), str(r.to_id))
            edges.append(
                {
                    "from_type": fr[0],
                    "from_id": fr[1],
                    "rel_type": str(r.rel_type),
                    "to_type": to[0],
                    "to_id": to[1],
                }
            )
            for key in (fr, to):
                if key not in nodes:
                    nodes[key] = {"entity_type": key[0], "id": key[1], "name_ar": None}
                    next_frontier.append(key)
        frontier = next_frontier

    return {
        "seeds": seed_nodes,
        "nodes": list(nodes.values()),
        "edges": edges,
    }


