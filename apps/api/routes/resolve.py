"""
Resolver inspection endpoints (Arabic-first).

These routes are for product-quality UX:
- show what entities the system detected and why
- support autocomplete / disambiguation flows
"""

from __future__ import annotations

from fastapi import APIRouter, Query
from pydantic import BaseModel
from sqlalchemy import text

from apps.api.core.database import get_session
from apps.api.retrieve.entity_resolver import EntityResolver

router = APIRouter()


class ResolveItem(BaseModel):
    type: str
    id: str
    name_ar: str
    match_type: str
    confidence: float


class ResolveResponse(BaseModel):
    query: str
    items: list[ResolveItem]


@router.get("/resolve", response_model=ResolveResponse)
async def resolve_entities(q: str = Query(..., description="Arabic query or concept term")):
    """
    Resolve entities from a query using the same resolver used in /ask LISTEN.
    """
    async with get_session() as session:
        resolver = EntityResolver()
        pillars = (await session.execute(text("SELECT id, name_ar FROM pillar"))).fetchall()
        core_values = (await session.execute(text("SELECT id, name_ar FROM core_value"))).fetchall()
        sub_values = (await session.execute(text("SELECT id, name_ar FROM sub_value"))).fetchall()
        resolver.load_entities(
            pillars=[{"id": r.id, "name_ar": r.name_ar} for r in pillars],
            core_values=[{"id": r.id, "name_ar": r.name_ar} for r in core_values],
            sub_values=[{"id": r.id, "name_ar": r.name_ar} for r in sub_values],
            aliases_path="data/static/aliases_ar.json",
        )
        results = resolver.resolve(q)
        items = [
            ResolveItem(
                type=r.entity_type.value,
                id=r.entity_id,
                name_ar=r.name_ar,
                match_type=r.match_type,
                confidence=r.confidence,
            )
            for r in results[:20]
        ]
        return ResolveResponse(query=q, items=items)


