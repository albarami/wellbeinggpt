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
from apps.api.graph.ref_coverage import ref_coverage
from apps.api.graph.explain import shortest_path
from apps.api.graph.impact import impact_propagation
from apps.api.graph.compare import compare_concepts

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


class RefCoveragePillar(BaseModel):
    pillar_id: str
    pillar_name_ar: str | None = None
    entity_count: int


class RefCoverageEntity(BaseModel):
    entity_type: str
    entity_id: str


class RefCoverageResponse(BaseModel):
    ref_node_id: str
    entities: list[RefCoverageEntity]
    pillars: list[RefCoveragePillar]


@router.get("/graph/ref/{ref_node_id}/coverage", response_model=RefCoverageResponse)
async def graph_ref_coverage(
    ref_node_id: str,
    limit: int = Query(default=200, ge=1, le=2000),
):
    """Coverage drill-down for a Quran/Hadith ref node."""
    async with get_session() as session:
        data = await ref_coverage(session, ref_node_id=ref_node_id, limit=limit)
        return RefCoverageResponse(
            ref_node_id=data["ref_node_id"],
            entities=[RefCoverageEntity(**e) for e in data["entities"]],
            pillars=[RefCoveragePillar(**p) for p in data["pillars"]],
        )


class PathNode(BaseModel):
    type: str
    id: str
    via_rel: str | None = None


class ExplainPathResponse(BaseModel):
    found: bool
    path: list[PathNode]


@router.get("/graph/explain/path", response_model=ExplainPathResponse)
async def graph_explain_path(
    start_type: str = Query(..., description="pillar|core_value|sub_value|evidence|ref"),
    start_id: str = Query(...),
    target_type: str = Query(...),
    target_id: str = Query(...),
    max_depth: int = Query(default=4, ge=1, le=10),
):
    """Explain connections by returning a shortest path over approved edges."""
    async with get_session() as session:
        res = await shortest_path(
            session,
            start_type=start_type,
            start_id=start_id,
            target_type=target_type,
            target_id=target_id,
            max_depth=max_depth,
            rel_types=["CONTAINS", "SUPPORTED_BY", "MENTIONS_REF", "REFERS_TO", "SHARES_REF", "SAME_NAME"],
        )
        return ExplainPathResponse(found=res["found"], path=[PathNode(**n) for n in res["path"]])


class ImpactItem(BaseModel):
    entity_type: str
    entity_id: str
    score: float
    reasons: list[str]
    shared_ref_nodes: list[str] = Field(default_factory=list)


class ImpactResponse(BaseModel):
    seed: dict
    items: list[ImpactItem]


@router.get("/graph/impact", response_model=ImpactResponse)
async def graph_impact(
    entity_type: str = Query(..., description="pillar|core_value|sub_value"),
    entity_id: str = Query(...),
    max_depth: int = Query(default=3, ge=1, le=6),
    top_k: int = Query(default=20, ge=1, le=200),
):
    """Deterministic impact propagation (related values across the framework)."""
    async with get_session() as session:
        data = await impact_propagation(
            session,
            entity_type=entity_type,
            entity_id=entity_id,
            max_depth=max_depth,
            top_k=top_k,
        )
        return ImpactResponse(seed=data["seed"], items=[ImpactItem(**i) for i in data["items"]])


class CompareSide(BaseModel):
    query: str
    entity: dict | None = None
    packets: list[dict] = Field(default_factory=list)


class CompareResponse(BaseModel):
    left: CompareSide
    right: CompareSide
    shared_refs: list[str]
    left_only_refs: list[str]
    right_only_refs: list[str]


@router.get("/graph/compare", response_model=CompareResponse)
async def graph_compare(
    q1: str = Query(..., description="First concept/value"),
    q2: str = Query(..., description="Second concept/value"),
    per_side_limit: int = Query(default=20, ge=1, le=100),
):
    """
    Evidence-only comparison for two concepts/values.

    This does NOT assert contradiction; it returns what the framework evidences for each side.
    """
    async with get_session() as session:
        data = await compare_concepts(session, q1=q1, q2=q2, per_side_limit=per_side_limit)
        return CompareResponse(
            left=CompareSide(**data["left"]),
            right=CompareSide(**data["right"]),
            shared_refs=data["shared_refs"],
            left_only_refs=data["left_only_refs"],
            right_only_refs=data["right_only_refs"],
        )


