"""
Deterministic structure answering for "list" intents.

This is the "shock" factor without hallucination:
- Always correct from DB structure
- Always cite heading chunks
- Works even if GPT‑5 is unavailable
"""

from __future__ import annotations

from typing import Any, Optional

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession


async def _pick_heading_chunk(
    session: AsyncSession,
    entity_type: str,
    entity_id: str,
    prefix: str,
) -> Optional[dict[str, Any]]:
    row = (
        await session.execute(
            text(
                """
                SELECT chunk_id, source_anchor, text_ar
                FROM chunk
                WHERE entity_type = :et
                  AND entity_id = :eid
                  AND chunk_type = 'definition'
                  AND text_ar LIKE :pfx
                ORDER BY chunk_id
                LIMIT 1
                """
            ),
            {"et": entity_type, "eid": entity_id, "pfx": f"{prefix}%"},
        )
    ).fetchone()
    if not row:
        return None
    return {"chunk_id": row.chunk_id, "source_anchor": row.source_anchor, "text_ar": row.text_ar}


async def answer_list_pillars(
    session: AsyncSession,
) -> Optional[dict[str, Any]]:
    pillars = (
        await session.execute(text("SELECT id::text AS id, name_ar FROM pillar ORDER BY id"))
    ).fetchall()
    if len(pillars) != 5:
        return None

    names = []
    citations = []
    for p in pillars:
        names.append(p.name_ar)
        ch = await _pick_heading_chunk(session, "pillar", p.id, "الركيزة:")
        if ch:
            citations.append({"chunk_id": ch["chunk_id"], "source_anchor": ch["source_anchor"], "ref": None})
    if not citations:
        return None
    return {
        "answer_ar": "ركائز الحياة الطيبة الخمس هي: " + "، ".join(names) + ".",
        "citations": citations[:5],
    }


async def answer_list_core_values_in_pillar(
    session: AsyncSession,
    pillar_id: str,
    pillar_name_ar: str,
) -> Optional[dict[str, Any]]:
    cvs = (
        await session.execute(
            text(
                """
                SELECT id::text AS id, name_ar
                FROM core_value
                WHERE pillar_id = :pid
                ORDER BY id
                """
            ),
            {"pid": pillar_id},
        )
    ).fetchall()
    if not cvs:
        return None

    names = []
    citations = []
    for cv in cvs:
        names.append(cv.name_ar)
        ch = await _pick_heading_chunk(session, "core_value", cv.id, "القيمة الكلية:")
        if ch:
            citations.append({"chunk_id": ch["chunk_id"], "source_anchor": ch["source_anchor"], "ref": None})

    if not citations:
        return None

    return {
        "answer_ar": f"القيم الكلية في {pillar_name_ar} هي: " + "، ".join(names) + ".",
        "citations": citations[:10],
    }


