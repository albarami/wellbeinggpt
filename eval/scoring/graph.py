"""Graph correctness scoring."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from apps.api.retrieve.normalize_ar import normalize_for_matching
from eval.types import EvalOutputRow


def _parse_edge_id(edge_id: str) -> Optional[dict[str, str]]:
    # edge_id format: "from_type:from_id::REL::to_type:to_id"
    if not edge_id or "::" not in edge_id:
        return None
    a, rel, b = edge_id.split("::", 2)
    if ":" not in a or ":" not in b:
        return None
    ft, _, fid = a.partition(":")
    tt, _, tid = b.partition(":")
    return {"from_type": ft, "from_id": fid, "rel_type": rel, "to_type": tt, "to_id": tid}


async def edge_exists(session: AsyncSession, edge_id: str) -> bool:
    p = _parse_edge_id(edge_id)
    if not p:
        return False
    row = (
        await session.execute(
            text(
                """
                SELECT 1
                FROM edge
                WHERE status='approved'
                  AND from_type=:ft AND from_id=:fid
                  AND rel_type=:rt
                  AND to_type=:tt AND to_id=:tid
                LIMIT 1
                """
            ),
            {
                "ft": p["from_type"],
                "fid": p["from_id"],
                "rt": p["rel_type"],
                "tt": p["to_type"],
                "tid": p["to_id"],
            },
        )
    ).fetchone()
    if row:
        return True

    # Allow reverse direction for undirected traversal semantics.
    row2 = (
        await session.execute(
            text(
                """
                SELECT 1
                FROM edge
                WHERE status='approved'
                  AND from_type=:tt AND from_id=:tid
                  AND rel_type=:rt
                  AND to_type=:ft AND to_id=:fid
                LIMIT 1
                """
            ),
            {
                "ft": p["from_type"],
                "fid": p["from_id"],
                "rt": p["rel_type"],
                "tt": p["to_type"],
                "tid": p["to_id"],
            },
        )
    ).fetchone()
    return bool(row2)


async def _semantic_edge_justified_by_citations(
    session: Optional[AsyncSession],
    *,
    edge_id: str,
    relation_type: Optional[str],
    cited_chunk_ids: list[str],
) -> bool:
    """
    Return True if a semantic edge has >=1 justification span and at least one
    justification span chunk is cited by the answer.
    """
    if session is None:
        return False

    p = _parse_edge_id(edge_id)
    if not p:
        return False

    # Resolve the DB edge UUID (either direction).
    where = """
        SELECT id
        FROM edge
        WHERE status='approved'
          AND from_type=:ft AND from_id=:fid
          AND rel_type=:rt
          AND to_type=:tt AND to_id=:tid
    """
    params: dict[str, Any] = {
        "ft": p["from_type"],
        "fid": p["from_id"],
        "rt": p["rel_type"],
        "tt": p["to_type"],
        "tid": p["to_id"],
    }
    if relation_type:
        where += " AND relation_type=:rel"
        params["rel"] = relation_type

    row = (await session.execute(text(where + " LIMIT 1"), params)).fetchone()
    if not row:
        # Reverse direction.
        params2 = {"ft": p["to_type"], "fid": p["to_id"], "rt": p["rel_type"], "tt": p["from_type"], "tid": p["from_id"]}
        if relation_type:
            params2["rel"] = relation_type
        row = (await session.execute(text(where + " LIMIT 1"), params2)).fetchone()
    if not row:
        return False

    edge_uuid = str(row.id)
    span_rows = (
        await session.execute(
            text(
                """
                SELECT chunk_id
                FROM edge_justification_span
                WHERE edge_id=:eid
                LIMIT 50
                """
            ),
            {"eid": edge_uuid},
        )
    ).fetchall()
    if not span_rows:
        return False

    just_chunk_ids = {str(r.chunk_id) for r in span_rows if getattr(r, "chunk_id", None)}
    return any(cid in just_chunk_ids for cid in (cited_chunk_ids or []))


@dataclass(frozen=True)
class GraphMetrics:
    path_valid_rate: float
    cross_pillar_hit_rate: float
    explanation_grounded_rate: float


async def score_graph(
    *,
    session: Optional[AsyncSession],
    outputs: list[EvalOutputRow],
    dataset_by_id: dict[str, dict[str, Any]],
) -> GraphMetrics:
    total_edges = 0
    valid_edges = 0
    cross_total = 0
    cross_hit = 0
    cross_grounded = 0

    for r in outputs:
        d = dataset_by_id.get(r.id, {})
        if str(d.get("type")) != "cross_pillar":
            continue

        cross_total += 1
        if r.graph_trace and (r.graph_trace.edges or r.graph_trace.paths):
            cross_hit += 1

        # Grounded explanation requirement (deterministic):
        # - must not abstain
        # - must provide path trace
        # - if justification is a ref_id like "quran:<ref>" or "hadith:<ref>",
        #   require at least one cited chunk to include that ref in chunk_ref.
        grounded = (not r.abstained) and bool(r.graph_trace.edges or r.graph_trace.paths)
        if grounded:
            req_paths = d.get("required_graph_paths") or []
            cited_chunk_ids = [c.source_id for c in (r.citations or []) if (c.source_id or "").strip()]
            quotes_norm = [normalize_for_matching(c.quote or "") for c in (r.citations or [])]
            for p in req_paths:
                just = str(p.get("justification") or "").strip()
                rel_type = str(p.get("rel_type") or "")
                relation_type = str(p.get("relation_type") or "").strip() or None
                if not just:
                    continue

                # Scholar semantic edge grounding: must have edge justification spans, and cite at least one.
                if rel_type == "SCHOLAR_LINK":
                    edges = p.get("edges") or []
                    eid = str(edges[0]) if (isinstance(edges, list) and edges) else ""
                    ok = await _semantic_edge_justified_by_citations(
                        session,
                        edge_id=eid,
                        relation_type=relation_type,
                        cited_chunk_ids=cited_chunk_ids,
                    )
                    if not ok:
                        grounded = False
                        break
                    continue

                # SHARES_REF justifications are stored as "type:ref" from chunk rows.
                if ":" in just and rel_type == "SHARES_REF":
                    ref_type, _, ref = just.partition(":")
                    ref_type = ref_type.strip()
                    ref = ref.strip()
                    if not (ref_type and ref):
                        grounded = False
                        break

                    # If no DB session is available (unit tests), fall back to deterministic
                    # quote containment check to detect citation shuffling/leakage.
                    if session is None:
                        jn = normalize_for_matching(just)
                        if not any(jn and (jn in q) for q in quotes_norm if q):
                            grounded = False
                            break
                        continue

                    found = False
                    for cid in cited_chunk_ids:
                        row_ref = (
                            await session.execute(
                                text(
                                    """
                                    SELECT 1
                                    FROM chunk_ref
                                    WHERE chunk_id=:cid AND ref_type=:rt AND ref=:r
                                    LIMIT 1
                                    """
                                ),
                                {"cid": cid, "rt": ref_type, "r": ref},
                            )
                        ).fetchone()
                        if row_ref:
                            found = True
                            break
                    if not found:
                        grounded = False
                        break

                # SAME_NAME is grounded by having evidence for both endpoints; we approximate by >=2 citations.
                elif rel_type == "SAME_NAME":
                    if len(cited_chunk_ids) < 2:
                        grounded = False
                        break
                else:
                    # Unknown rel_type or missing rel_type: require at least one justification token
                    # to appear in some citation quote (fallback, deterministic).
                    jn = normalize_for_matching(just)
                    if jn and not any(jn in q for q in quotes_norm if q):
                        grounded = False
                        break

        if grounded:
            cross_grounded += 1

        # Validate required edges if present
        req_paths = d.get("required_graph_paths") or []
        for p in req_paths:
            edges = p.get("edges") or []
            if not isinstance(edges, list):
                continue
            for eid in edges:
                eid = str(eid)
                total_edges += 1
                if session is not None and await edge_exists(session, eid):
                    valid_edges += 1

    path_valid_rate = (valid_edges / total_edges) if total_edges else 1.0
    cross_hit_rate = (cross_hit / cross_total) if cross_total else 1.0
    grounded_rate = (cross_grounded / cross_total) if cross_total else 1.0

    return GraphMetrics(
        path_valid_rate=path_valid_rate,
        cross_pillar_hit_rate=cross_hit_rate,
        explanation_grounded_rate=grounded_rate,
    )
