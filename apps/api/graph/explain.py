"""
Explainable graph paths.

Provides deterministic, citeable "how are these connected?" paths
without any LLM involvement.
"""

from __future__ import annotations

from collections import deque
from typing import Any, Iterable, Optional

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession


AllowedRel = tuple[str, ...]


async def _neighbors(
    session: AsyncSession,
    node_type: str,
    node_id: str,
    rel_types: Optional[list[str]] = None,
) -> list[dict[str, str]]:
    params: dict[str, Any] = {"t": node_type, "i": node_id}
    filt = ""
    if rel_types:
        filt = " AND e.rel_type = ANY(:rels)"
        params["rels"] = rel_types

    rows = (
        await session.execute(
            text(
                f"""
                SELECT e.rel_type, e.to_type AS n_type, e.to_id AS n_id
                FROM edge e
                WHERE e.status='approved'
                  AND e.from_type=:t AND e.from_id=:i
                  {filt}
                UNION ALL
                SELECT e.rel_type, e.from_type AS n_type, e.from_id AS n_id
                FROM edge e
                WHERE e.status='approved'
                  AND e.to_type=:t AND e.to_id=:i
                  {filt}
                """
            ),
            params,
        )
    ).fetchall()

    return [{"rel_type": str(r.rel_type), "node_type": str(r.n_type), "node_id": str(r.n_id)} for r in rows]


async def shortest_path(
    session: AsyncSession,
    start_type: str,
    start_id: str,
    target_type: str,
    target_id: str,
    max_depth: int = 4,
    rel_types: Optional[list[str]] = None,
) -> dict[str, Any]:
    """
    Find a shortest path between two nodes using BFS over approved edges.

    Returns:
        {found: bool, path: [{type,id,via_rel}]}
    """
    start = (start_type, start_id)
    target = (target_type, target_id)
    if start == target:
        return {"found": True, "path": [{"type": start_type, "id": start_id, "via_rel": None}]}

    q: deque[tuple[str, str]] = deque([start])
    prev: dict[tuple[str, str], tuple[tuple[str, str], str]] = {}
    visited: set[tuple[str, str]] = {start}

    depth = 0
    while q and depth <= max_depth:
        for _ in range(len(q)):
            cur_t, cur_i = q.popleft()
            for n in await _neighbors(session, cur_t, cur_i, rel_types=rel_types):
                nxt = (n["node_type"], n["node_id"])
                if nxt in visited:
                    continue
                visited.add(nxt)
                prev[nxt] = ((cur_t, cur_i), n["rel_type"])
                if nxt == target:
                    # Reconstruct
                    path_nodes: list[tuple[str, str, Optional[str]]] = []
                    node = target
                    via: Optional[str] = None
                    while node != start:
                        pnode, rel = prev[node]
                        path_nodes.append((node[0], node[1], rel))
                        node = pnode
                    path_nodes.append((start[0], start[1], None))
                    path_nodes.reverse()
                    return {
                        "found": True,
                        "path": [{"type": t, "id": i, "via_rel": r} for t, i, r in path_nodes],
                    }
                q.append(nxt)
        depth += 1

    return {"found": False, "path": []}


