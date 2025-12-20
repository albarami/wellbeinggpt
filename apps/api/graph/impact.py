"""
Impact propagation over the graph (deterministic).

Given a starting entity, find the most related entities across pillars
based on:
- shared ref nodes (MENTIONS_REF)
- SAME_NAME edges
- short graph distance
"""

from __future__ import annotations

from collections import defaultdict, deque
from typing import Any, Optional

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession


async def impact_propagation(
    session: AsyncSession,
    entity_type: str,
    entity_id: str,
    max_depth: int = 3,
    top_k: int = 20,
) -> dict[str, Any]:
    """
    Compute related entities with an explainable score.

    Returns:
        {
          "seed": {...},
          "items": [{"entity_type","entity_id","score","reasons":[...]}]
        }
    """
    # 1) Collect seed ref nodes
    seed_refs = (
        await session.execute(
            text(
                """
                SELECT to_id AS ref_node_id
                FROM edge
                WHERE status='approved'
                  AND rel_type='MENTIONS_REF'
                  AND from_type=:t AND from_id=:i
                  AND to_type='ref'
                """
            ),
            {"t": entity_type, "i": entity_id},
        )
    ).fetchall()
    seed_ref_ids = {str(r.ref_node_id) for r in seed_refs}

    # 2) Candidates: entities that mention any of these refs
    candidates = []
    if seed_ref_ids:
        cand_rows = (
            await session.execute(
                text(
                    """
                    SELECT from_type AS entity_type, from_id AS entity_id, to_id AS ref_node_id
                    FROM edge
                    WHERE status='approved'
                      AND rel_type='MENTIONS_REF'
                      AND to_type='ref'
                      AND to_id = ANY(:refs)
                    """
                ),
                {"refs": list(seed_ref_ids)},
            )
        ).fetchall()
        candidates = [dict(r._mapping) for r in cand_rows]

    # 3) Score by shared ref count, and add SAME_NAME bonus.
    shared_counts: dict[tuple[str, str], int] = defaultdict(int)
    shared_refs: dict[tuple[str, str], set[str]] = defaultdict(set)
    for c in candidates:
        key = (str(c["entity_type"]), str(c["entity_id"]))
        if key == (entity_type, entity_id):
            continue
        shared_counts[key] += 1
        shared_refs[key].add(str(c["ref_node_id"]))

    # SAME_NAME bonus for sub_value only
    same_name_bonus: dict[tuple[str, str], int] = defaultdict(int)
    if entity_type == "sub_value":
        rows = (
            await session.execute(
                text(
                    """
                    SELECT
                        CASE WHEN from_id=:i THEN to_id ELSE from_id END AS other_id
                    FROM edge
                    WHERE status='approved'
                      AND rel_type='SAME_NAME'
                      AND ((from_type='sub_value' AND from_id=:i) OR (to_type='sub_value' AND to_id=:i))
                    """
                ),
                {"i": entity_id},
            )
        ).fetchall()
        for r in rows:
            same_name_bonus[("sub_value", str(r.other_id))] = 2

    # 4) Distance penalty: BFS over approved edges (limited relationship set)
    # We keep it simple and deterministic.
    dist: dict[tuple[str, str], int] = {(entity_type, entity_id): 0}
    q: deque[tuple[str, str]] = deque([(entity_type, entity_id)])
    rels = ["CONTAINS", "SUPPORTED_BY", "MENTIONS_REF", "REFERS_TO", "SHARES_REF", "SAME_NAME"]

    while q:
        cur = q.popleft()
        d = dist[cur]
        if d >= max_depth:
            continue
        rows = (
            await session.execute(
                text(
                    """
                    SELECT
                        CASE
                            WHEN from_type=:t AND from_id=:i THEN to_type
                            ELSE from_type
                        END AS n_type,
                        CASE
                            WHEN from_type=:t AND from_id=:i THEN to_id
                            ELSE from_id
                        END AS n_id
                    FROM edge
                    WHERE status='approved'
                      AND rel_type = ANY(:rels)
                      AND (
                        (from_type=:t AND from_id=:i)
                        OR
                        (to_type=:t AND to_id=:i)
                      )
                    """
                ),
                {"t": cur[0], "i": cur[1], "rels": rels},
            )
        ).fetchall()
        for r in rows:
            nxt = (str(r.n_type), str(r.n_id))
            if nxt not in dist:
                dist[nxt] = d + 1
                q.append(nxt)

    items = []
    for key, cnt in shared_counts.items():
        # Score: shared refs (strong) + same-name bonus + distance discount
        d = dist.get(key, max_depth + 1)
        score = float(cnt) + float(same_name_bonus.get(key, 0)) + (0.5 / max(d, 1))
        reasons = [f"shared_refs={cnt}"]
        if same_name_bonus.get(key):
            reasons.append("same_name=2")
        if d <= max_depth:
            reasons.append(f"graph_distance={d}")
        items.append(
            {
                "entity_type": key[0],
                "entity_id": key[1],
                "score": score,
                "reasons": reasons,
                "shared_ref_nodes": sorted(list(shared_refs[key]))[:10],
            }
        )

    items.sort(key=lambda x: x["score"], reverse=True)
    return {"seed": {"entity_type": entity_type, "entity_id": entity_id}, "items": items[:top_k]}






