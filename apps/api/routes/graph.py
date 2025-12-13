"""
Graph analytics routes.

These endpoints are meant to power UI capabilities like:
- "Patience network" concept graphs
- Central Quran/Hadith verse ranking
- Cross-pillar discovery via ref nodes
"""

from __future__ import annotations

from fastapi import APIRouter, Query
from pydantic import BaseModel, Field

from apps.api.core.database import get_session
from apps.api.graph.analytics import concept_network, top_ref_nodes

router = APIRouter()


class RefCentralityItem(BaseModel):
    ref_node_id: str
    entity_count: int
    evidence_count: int


class RefCentralityResponse(BaseModel):
    items: list[RefCentralityItem]


@router.get("/graph/refs/centrality", response_model=RefCentralityResponse)
async def refs_centrality(
    evidence_type: str | None = Query(default=None, description="quran|hadith|book"),
    top_k: int = Query(default=10, ge=1, le=100),
):
    """
    Rank Quran/Hadith refs by graph centrality (how many values they support).
    """
    async with get_session() as session:
        items = await top_ref_nodes(session, evidence_type=evidence_type, top_k=top_k)
        return RefCentralityResponse(items=[RefCentralityItem(**i) for i in items])


class NetworkEdge(BaseModel):
    from_type: str
    from_id: str
    rel_type: str
    to_type: str
    to_id: str


class NetworkNode(BaseModel):
    entity_type: str
    id: str
    name_ar: str | None = None


class ConceptNetworkResponse(BaseModel):
    seeds: list[NetworkNode]
    nodes: list[NetworkNode]
    edges: list[NetworkEdge]


@router.get("/graph/network", response_model=ConceptNetworkResponse)
async def graph_network(
    q: str = Query(..., description="Arabic concept/value name; e.g., الصبر"),
    depth: int = Query(default=2, ge=1, le=5),
    limit_entities: int = Query(default=20, ge=1, le=200),
):
    """
    Build a small concept graph around entities matching q (for visualization).
    """
    async with get_session() as session:
        data = await concept_network(session, name_pattern=q, depth=depth, limit_entities=limit_entities)
        return ConceptNetworkResponse(
            seeds=[NetworkNode(**n) for n in data["seeds"]],
            nodes=[NetworkNode(**n) for n in data["nodes"]],
            edges=[NetworkEdge(**e) for e in data["edges"]],
        )


