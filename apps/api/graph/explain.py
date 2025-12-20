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
    require_grounded_semantic: bool = True,
) -> list[dict[str, Any]]:
    params: dict[str, Any] = {"t": node_type, "i": node_id}
    filt = ""
    if rel_types:
        filt = " AND e.rel_type = ANY(:rels)"
        params["rels"] = rel_types

    rows = (
        await session.execute(
            text(
                f"""
                SELECT
                  e.id::text AS edge_id,
                  e.rel_type,
                  e.relation_type,
                  e.to_type AS n_type,
                  e.to_id AS n_id,
                  CASE
                    WHEN e.relation_type IS NULL THEN TRUE
                    WHEN EXISTS (SELECT 1 FROM edge_justification_span s WHERE s.edge_id=e.id) THEN TRUE
                    ELSE FALSE
                  END AS is_grounded
                FROM edge e
                WHERE e.status='approved'
                  AND e.from_type=:t AND e.from_id=:i
                  {filt}
                UNION ALL
                SELECT
                  e.id::text AS edge_id,
                  e.rel_type,
                  e.relation_type,
                  e.from_type AS n_type,
                  e.from_id AS n_id,
                  CASE
                    WHEN e.relation_type IS NULL THEN TRUE
                    WHEN EXISTS (SELECT 1 FROM edge_justification_span s WHERE s.edge_id=e.id) THEN TRUE
                    ELSE FALSE
                  END AS is_grounded
                FROM edge e
                WHERE e.status='approved'
                  AND e.to_type=:t AND e.to_id=:i
                  {filt}
                """
            ),
            params,
        )
    ).fetchall()

    out: list[dict[str, str]] = []
    for r in rows:
        rel_type = str(r.rel_type)
        relation_type = (str(r.relation_type) if r.relation_type is not None else None)
        grounded = bool(getattr(r, "is_grounded", True))

        # Hard gate: semantic edges are only eligible if edge justification spans exist.
        if require_grounded_semantic and relation_type and (not grounded):
            continue

        out.append(
            {
                "edge_id": str(r.edge_id),
                "rel_type": rel_type,
                "relation_type": relation_type or "",
                "node_type": str(r.n_type),
                "node_id": str(r.n_id),
                "is_grounded": "1" if grounded else "0",
            }
        )
    return out


async def shortest_path(
    session: AsyncSession,
    start_type: str,
    start_id: str,
    target_type: str,
    target_id: str,
    max_depth: int = 4,
    rel_types: Optional[list[str]] = None,
    require_grounded_semantic: bool = True,
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
    # prev[next] = (prev_node, edge_meta)
    prev: dict[tuple[str, str], tuple[tuple[str, str], dict[str, Any]]] = {}
    visited: set[tuple[str, str]] = {start}

    # depth here counts hops already taken from the start node.
    # We expand one BFS layer per loop; each layer adds exactly 1 hop.
    depth = 0
    while q and depth < max_depth:
        for _ in range(len(q)):
            cur_t, cur_i = q.popleft()
            for n in await _neighbors(
                session,
                cur_t,
                cur_i,
                rel_types=rel_types,
                require_grounded_semantic=require_grounded_semantic,
            ):
                nxt = (n["node_type"], n["node_id"])
                if nxt in visited:
                    continue
                visited.add(nxt)
                prev[nxt] = ((cur_t, cur_i), n)
                if nxt == target:
                    # Reconstruct
                    path_nodes: list[tuple[str, str, Optional[dict[str, Any]]]] = []
                    node = target
                    while node != start:
                        pnode, edge_meta = prev[node]
                        path_nodes.append((node[0], node[1], edge_meta))
                        node = pnode
                    path_nodes.append((start[0], start[1], None))
                    path_nodes.reverse()

                    # Attach justification spans for edges (best-effort).
                    edge_ids = []
                    for _, _, meta in path_nodes:
                        if meta and meta.get("edge_id"):
                            edge_ids.append(str(meta.get("edge_id")))
                    spans_by_edge: dict[str, list[dict[str, Any]]] = {}
                    try:
                        if edge_ids:
                            span_rows = (
                                await session.execute(
                                    text(
                                        """
                                        SELECT edge_id::text AS edge_id, chunk_id, span_start, span_end, quote
                                        FROM edge_justification_span
                                        WHERE edge_id::text = ANY(:eids)
                                        ORDER BY edge_id, chunk_id, span_start
                                        """
                                    ),
                                    {"eids": edge_ids},
                                )
                            ).fetchall()
                            for r in span_rows:
                                spans_by_edge.setdefault(str(r.edge_id), []).append(
                                    {
                                        "chunk_id": str(r.chunk_id),
                                        "span_start": int(r.span_start),
                                        "span_end": int(r.span_end),
                                        "quote": str(r.quote),
                                    }
                                )
                    except Exception:
                        spans_by_edge = {}

                    return {
                        "found": True,
                        "path": [
                            {
                                "type": t,
                                "id": i,
                                "via_rel": (meta.get("rel_type") if meta else None),
                                "edge_id": (meta.get("edge_id") if meta else None),
                                "relation_type": (meta.get("relation_type") if meta else None),
                                "justification_spans": (
                                    spans_by_edge.get(str(meta.get("edge_id")), []) if meta and meta.get("edge_id") else []
                                ),
                            }
                            for t, i, meta in path_nodes
                        ],
                    }
                q.append(nxt)
        depth += 1

    return {"found": False, "path": []}


