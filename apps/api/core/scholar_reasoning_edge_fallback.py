"""Deterministic semantic-edge fallback for scholar reasoning.

Reason: keep `scholar_reasoning_impl.py` under 500 LOC while still supporting
cross-pillar questions that mention pillars but do not name entities.
"""

from __future__ import annotations

from typing import Any

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from apps.api.retrieve.graph_retriever import get_entity_neighbors


# Deterministic pillar keyword mapping (Arabic-first).
_PILLAR_KEYWORDS: dict[str, str] = {
    # P001
    "روحي": "P001",
    "الروحية": "P001",
    "الحياة الروحية": "P001",
    # P002
    "عاطفي": "P002",
    "العاطفية": "P002",
    "الحياة العاطفية": "P002",
    # P003
    "فكري": "P003",
    "الفكرية": "P003",
    "الحياة الفكرية": "P003",
    # P004
    "بدني": "P004",
    "البدنية": "P004",
    "الجسد": "P004",
    "الجسم": "P004",
    "جسم": "P004",
    "الحياة البدنية": "P004",
    # P005
    "اجتماعي": "P005",
    "الاجتماعية": "P005",
    "الحياة الاجتماعية": "P005",
}


def detect_pillar_ids_from_question(*, question_norm: str) -> list[str]:
    """Extract pillar IDs from Arabic keywords deterministically."""

    q = question_norm or ""
    found: list[str] = []
    for k, pid in _PILLAR_KEYWORDS.items():
        if k and (k in q) and (pid not in found):
            found.append(pid)
    return found[:3]


async def pillar_id_for_entity(*, session: AsyncSession, entity_type: str, entity_id: str) -> str | None:
    """Resolve an entity to its pillar_id (best-effort)."""

    et = (entity_type or "").strip()
    eid = (entity_id or "").strip()
    if not et or not eid:
        return None
    if et == "pillar":
        return eid
    if et == "core_value":
        row = (
            await session.execute(
                text("SELECT pillar_id FROM core_value WHERE id = :id LIMIT 1"),
                {"id": eid},
            )
        ).fetchone()
        return str(row.pillar_id) if row and row.pillar_id else None
    if et == "sub_value":
        row = (
            await session.execute(
                text(
                    """
                    SELECT cv.pillar_id AS pillar_id
                    FROM sub_value sv
                    JOIN core_value cv ON cv.id = sv.core_value_id
                    WHERE sv.id = :id
                    LIMIT 1
                    """
                ),
                {"id": eid},
            )
        ).fetchone()
        return str(row.pillar_id) if row and row.pillar_id else None
    return None


async def find_hub_entity(*, session: AsyncSession, min_distinct_pillars: int = 3) -> tuple[str, str] | None:
    """
    Find a deterministic "hub" entity (type,id) that links across many pillars.
    """

    rows = (
        await session.execute(
            text(
                """
                SELECT e.from_type, e.from_id, e.to_type, e.to_id
                FROM edge e
                WHERE e.rel_type = 'SCHOLAR_LINK'
                  AND e.status = 'approved'
                  AND e.relation_type IS NOT NULL
                  AND EXISTS (SELECT 1 FROM edge_justification_span s WHERE s.edge_id = e.id)
                ORDER BY COALESCE(e.strength_score, 0) DESC
                LIMIT 300
                """
            )
        )
    ).fetchall()

    if not rows:
        return None

    neighbor_pillars: dict[tuple[str, str], set[str]] = {}
    for r in rows:
        ft = str(r.from_type)
        fid = str(r.from_id)
        tt = str(r.to_type)
        tid = str(r.to_id)
        p_to = await pillar_id_for_entity(session=session, entity_type=tt, entity_id=tid)
        p_from = await pillar_id_for_entity(session=session, entity_type=ft, entity_id=fid)
        if p_to:
            neighbor_pillars.setdefault((ft, fid), set()).add(p_to)
        if p_from:
            neighbor_pillars.setdefault((tt, tid), set()).add(p_from)

    best = None
    best_n = -1
    for k, ps in neighbor_pillars.items():
        n = len(ps)
        if n >= min_distinct_pillars and n > best_n:
            best = k
            best_n = n
    return best


async def semantic_edges_fallback(
    *,
    session: AsyncSession,
    question_norm: str,
    max_edges: int,
    extra: bool = False,
) -> list[dict[str, Any]]:
    """
    Fallback semantic edge selection when entity resolution yields no usable anchors.
    """

    pillar_ids = detect_pillar_ids_from_question(question_norm=question_norm)
    if len(pillar_ids) >= 2:
        a, b = pillar_ids[0], pillar_ids[1]
        out: list[dict[str, Any]] = []
        for anchor in [a, b]:
            neigh = await get_entity_neighbors(
                session,
                "pillar",
                anchor,
                relationship_types=["SCHOLAR_LINK"],
                direction="both",
                status="approved",
            )
            for n in neigh:
                # Annotate source endpoint for downstream used-edge tracing.
                n["source_type"] = "pillar"
                n["source_id"] = anchor
                rt = str(n.get("relation_type") or "")
                spans = n.get("justification_spans") or []
                if not (rt and spans):
                    continue
                np = await pillar_id_for_entity(
                    session=session,
                    entity_type=str(n.get("neighbor_type") or ""),
                    entity_id=str(n.get("neighbor_id") or ""),
                )
                if not np:
                    continue
                if {anchor, np} == {a, b}:
                    out.append(n)
                if len(out) >= max_edges:
                    return out
        if out:
            return out[:max_edges]

    hub = await find_hub_entity(session=session, min_distinct_pillars=3)
    if not hub:
        return []
    hub_type, hub_id = hub
    neigh = await get_entity_neighbors(
        session,
        hub_type,
        hub_id,
        relationship_types=["SCHOLAR_LINK"],
        direction="both",
        status="approved",
    )
    out2: list[dict[str, Any]] = []
    seen_pillars: set[str] = set()
    for n in neigh:
        # Annotate source endpoint for downstream used-edge tracing.
        n["source_type"] = str(hub_type)
        n["source_id"] = str(hub_id)
        rt = str(n.get("relation_type") or "")
        spans = n.get("justification_spans") or []
        if not (rt and spans):
            continue
        np = await pillar_id_for_entity(
            session=session,
            entity_type=str(n.get("neighbor_type") or ""),
            entity_id=str(n.get("neighbor_id") or ""),
        )
        if not np:
            continue
        if np in seen_pillars:
            continue
        seen_pillars.add(np)
        out2.append(n)
        if len(out2) >= max_edges:
            break

    return out2[:max_edges]

