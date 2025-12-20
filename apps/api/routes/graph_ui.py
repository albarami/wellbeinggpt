"""
Graph explorer routes for the production UI (additive).

Reason:
- Keep apps/api/routes/graph.py under 500 LOC (repo rule).
- Provide UI-oriented endpoints with hard caps and grounded-only defaults.
"""

from __future__ import annotations

from collections import deque

from fastapi import APIRouter, Query
from pydantic import BaseModel, Field
from sqlalchemy import text

from apps.api.core.database import get_session
from apps.api.graph.explain import shortest_path

router = APIRouter()


class PathNode(BaseModel):
    type: str
    id: str
    via_rel: str | None = None
    edge_id: str | None = None
    relation_type: str | None = None
    justification_spans: list[dict] = Field(default_factory=list)


class ExplainPathResponse(BaseModel):
    found: bool
    path: list[PathNode]


@router.get("/graph/path", response_model=ExplainPathResponse)
async def graph_path(
    start_type: str = Query(..., description="pillar|core_value|sub_value|evidence|ref"),
    start_id: str = Query(...),
    target_type: str = Query(...),
    target_id: str = Query(...),
    max_depth: int = Query(default=4, ge=1, le=10),
):
    """Grounded path explanation (UI alias)."""
    async with get_session() as session:
        res = await shortest_path(
            session,
            start_type=start_type,
            start_id=start_id,
            target_type=target_type,
            target_id=target_id,
            max_depth=max_depth,
            rel_types=[
                "CONTAINS",
                "SUPPORTED_BY",
                "MENTIONS_REF",
                "REFERS_TO",
                "SHARES_REF",
                "SAME_NAME",
                "SCHOLAR_LINK",
            ],
            require_grounded_semantic=True,
        )
        return ExplainPathResponse(found=res["found"], path=[PathNode(**n) for n in res["path"]])


class EdgeEvidenceSpan(BaseModel):
    chunk_id: str
    span_start: int
    span_end: int
    quote: str
    source_doc_id: str | None = None
    source_anchor: str | None = None


class EdgeEvidenceResponse(BaseModel):
    edge_id: str
    spans: list[EdgeEvidenceSpan]


@router.get("/graph/edge/{edge_id}/evidence", response_model=EdgeEvidenceResponse)
async def graph_edge_evidence(edge_id: str):
    """Return the grounded justification spans for an edge."""
    async with get_session() as session:
        rows = (
            await session.execute(
                text(
                    """
                    SELECT
                      s.chunk_id,
                      s.span_start,
                      s.span_end,
                      s.quote,
                      c.source_doc_id::text AS source_doc_id,
                      c.source_anchor
                    FROM edge_justification_span s
                    LEFT JOIN chunk c ON c.chunk_id = s.chunk_id
                    WHERE s.edge_id::text = :eid
                    ORDER BY s.chunk_id, s.span_start
                    """
                ),
                {"eid": str(edge_id)},
            )
        ).fetchall()
        spans = [
            EdgeEvidenceSpan(
                chunk_id=str(r.chunk_id),
                span_start=int(r.span_start),
                span_end=int(r.span_end),
                quote=str(r.quote),
                source_doc_id=str(r.source_doc_id) if getattr(r, "source_doc_id", None) else None,
                source_anchor=str(r.source_anchor) if getattr(r, "source_anchor", None) else None,
            )
            for r in (rows or [])
        ]
        return EdgeEvidenceResponse(edge_id=str(edge_id), spans=spans)


class ExpandNode(BaseModel):
    node_type: str
    node_id: str
    label_ar: str | None = None
    pillar_id: str | None = None


class ExpandEdge(BaseModel):
    edge_id: str
    from_type: str
    from_id: str
    to_type: str
    to_id: str
    rel_type: str
    relation_type: str | None = None
    has_evidence: bool = False


class GraphExpandResponse(BaseModel):
    seed: ExpandNode
    nodes: list[ExpandNode]
    edges: list[ExpandEdge]
    grounded_only: bool
    truncated: bool
    total_nodes: int
    total_edges: int
    returned_nodes: int
    returned_edges: int


async def _hydrate_nodes(session, nodes: list[tuple[str, str]]) -> dict[tuple[str, str], dict]:
    """
    Hydrate node labels + pillar IDs for known entity types.

    Returns map[(type,id)] -> {label_ar, pillar_id}.
    """
    out: dict[tuple[str, str], dict] = {}
    by_type: dict[str, list[str]] = {}
    for t, i in nodes:
        by_type.setdefault(str(t), []).append(str(i))

    if by_type.get("pillar"):
        rows = (
            await session.execute(
                text("SELECT id, name_ar FROM pillar WHERE id = ANY(:ids)"),
                {"ids": sorted(list(set(by_type["pillar"])))},
            )
        ).fetchall()
        for r in rows:
            out[("pillar", str(r.id))] = {"label_ar": str(r.name_ar), "pillar_id": str(r.id)}

    if by_type.get("core_value"):
        rows = (
            await session.execute(
                text("SELECT id, name_ar, pillar_id FROM core_value WHERE id = ANY(:ids)"),
                {"ids": sorted(list(set(by_type["core_value"])))},
            )
        ).fetchall()
        for r in rows:
            out[("core_value", str(r.id))] = {"label_ar": str(r.name_ar), "pillar_id": str(r.pillar_id)}

    if by_type.get("sub_value"):
        rows = (
            await session.execute(
                text(
                    """
                    SELECT sv.id, sv.name_ar, cv.pillar_id
                    FROM sub_value sv
                    JOIN core_value cv ON cv.id = sv.core_value_id
                    WHERE sv.id = ANY(:ids)
                    """
                ),
                {"ids": sorted(list(set(by_type["sub_value"])))},
            )
        ).fetchall()
        for r in rows:
            out[("sub_value", str(r.id))] = {"label_ar": str(r.name_ar), "pillar_id": str(r.pillar_id)}

    return out


@router.get("/graph/expand", response_model=GraphExpandResponse)
async def graph_expand(
    node_type: str = Query(..., description="pillar|core_value|sub_value|evidence|ref"),
    node_id: str = Query(...),
    depth: int = Query(default=2, ge=1, le=5),
    rel_types: list[str] | None = Query(default=None, description="Structural rel_type filter (optional)"),
    relation_types: list[str] | None = Query(default=None, description="Semantic relation_type filter (optional)"),
    grounded_only: bool = Query(default=True, description="Default true: only grounded semantic edges"),
    max_nodes: int = Query(default=2000, ge=50, le=2000),
    max_edges: int = Query(default=5000, ge=50, le=5000),
):
    """
    Progressive subgraph expansion for Sigma.js UI.

    Safety/perf:
    - grounded_only defaults true (semantic edges must have evidence)
    - hard caps on nodes/edges (server-enforced)
    - deterministic traversal order and truncation
    """
    seed_key = (str(node_type), str(node_id))

    async with get_session() as session:
        visited: set[tuple[str, str]] = {seed_key}
        nodes: dict[tuple[str, str], ExpandNode] = {
            seed_key: ExpandNode(node_type=seed_key[0], node_id=seed_key[1])
        }
        edges: dict[str, ExpandEdge] = {}

        q: deque[tuple[tuple[str, str], int]] = deque([(seed_key, 0)])
        truncated = False
        total_edges_seen = 0

        while q:
            (cur_t, cur_i), d = q.popleft()
            if d >= depth or truncated:
                continue

            params = {"t": cur_t, "i": cur_i}
            filt = ""
            if rel_types:
                filt += " AND e.rel_type = ANY(:rel_types)"
                params["rel_types"] = rel_types
            if relation_types:
                filt += " AND e.relation_type = ANY(:relation_types)"
                params["relation_types"] = relation_types
            if grounded_only:
                filt += """
                AND (
                  e.relation_type IS NULL
                  OR EXISTS (SELECT 1 FROM edge_justification_span s WHERE s.edge_id=e.id)
                )
                """

            rows = (
                await session.execute(
                    text(
                        f"""
                        SELECT
                          e.id::text AS edge_id,
                          e.from_type, e.from_id,
                          e.to_type, e.to_id,
                          e.rel_type,
                          e.relation_type,
                          EXISTS (SELECT 1 FROM edge_justification_span s WHERE s.edge_id=e.id) AS has_evidence
                        FROM edge e
                        WHERE e.status='approved'
                          AND (
                            (e.from_type=:t AND e.from_id=:i)
                            OR (e.to_type=:t AND e.to_id=:i)
                          )
                          {filt}
                        ORDER BY e.id::text
                        """
                    ),
                    params,
                )
            ).fetchall()

            for r in rows:
                total_edges_seen += 1
                eid = str(r.edge_id)
                if eid not in edges:
                    if len(edges) >= max_edges:
                        truncated = True
                        break
                    edges[eid] = ExpandEdge(
                        edge_id=eid,
                        from_type=str(r.from_type),
                        from_id=str(r.from_id),
                        to_type=str(r.to_type),
                        to_id=str(r.to_id),
                        rel_type=str(r.rel_type),
                        relation_type=str(r.relation_type) if r.relation_type is not None else None,
                        has_evidence=bool(getattr(r, "has_evidence", False)),
                    )

                a = (str(r.from_type), str(r.from_id))
                b = (str(r.to_type), str(r.to_id))
                nxt = b if (cur_t, cur_i) == a else a

                if nxt not in visited:
                    if len(nodes) >= max_nodes:
                        truncated = True
                        break
                    visited.add(nxt)
                    nodes[nxt] = ExpandNode(node_type=nxt[0], node_id=nxt[1])
                    q.append((nxt, d + 1))

            if len(nodes) >= max_nodes or len(edges) >= max_edges:
                truncated = True

        meta = await _hydrate_nodes(session, sorted(list(nodes.keys())))
        for k, n in nodes.items():
            m = meta.get(k)
            if m:
                n.label_ar = m.get("label_ar")
                n.pillar_id = m.get("pillar_id")
            else:
                n.pillar_id = None

        seed_node = nodes[seed_key]
        node_list = [nodes[k] for k in sorted(nodes.keys())]
        edge_list = [edges[k] for k in sorted(edges.keys())]

        return GraphExpandResponse(
            seed=seed_node,
            nodes=node_list,
            edges=edge_list,
            grounded_only=bool(grounded_only),
            truncated=bool(truncated),
            total_nodes=len(visited),
            total_edges=int(total_edges_seen),
            returned_nodes=len(node_list),
            returned_edges=len(edge_list),
        )

