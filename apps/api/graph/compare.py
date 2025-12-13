"""
Evidence-only comparison utilities.

This powers "side-by-side" scholar workflows without claiming contradictions:
- compare two concepts/values
- show shared refs + distinct refs
- return citeable chunks for each side
"""

from __future__ import annotations

from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from apps.api.core.schemas import EntityType
from apps.api.retrieve.sql_retriever import get_chunks_with_refs, search_entities_by_name


async def compare_concepts(
    session: AsyncSession,
    q1: str,
    q2: str,
    per_side_limit: int = 20,
) -> dict[str, Any]:
    """
    Compare two queries by resolving to best-matching entities and returning their evidence packets.

    Failure behavior:
      If either side resolves to no entities, return empty side (caller can handle as 404/empty).
    """
    e1 = await search_entities_by_name(session, q1, limit=5)
    e2 = await search_entities_by_name(session, q2, limit=5)

    def _pick_first(ents: list[dict[str, Any]]) -> dict[str, Any] | None:
        if not ents:
            return None
        return ents[0]

    a = _pick_first(e1)
    b = _pick_first(e2)

    packets_a: list[dict[str, Any]] = []
    packets_b: list[dict[str, Any]] = []
    if a:
        packets_a = await get_chunks_with_refs(session, EntityType(a["entity_type"]), a["id"], limit=per_side_limit)
    if b:
        packets_b = await get_chunks_with_refs(session, EntityType(b["entity_type"]), b["id"], limit=per_side_limit)

    def _refs(packets: list[dict[str, Any]]) -> set[str]:
        out: set[str] = set()
        for p in packets:
            for r in p.get("refs", []) or []:
                if isinstance(r, dict) and r.get("type") and r.get("ref"):
                    out.add(f"{r['type']}:{r['ref']}")
        return out

    refs_a = _refs(packets_a)
    refs_b = _refs(packets_b)

    return {
        "left": {"query": q1, "entity": a, "packets": packets_a},
        "right": {"query": q2, "entity": b, "packets": packets_b},
        "shared_refs": sorted(list(refs_a.intersection(refs_b))),
        "left_only_refs": sorted(list(refs_a - refs_b)),
        "right_only_refs": sorted(list(refs_b - refs_a)),
    }


