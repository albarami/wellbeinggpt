"""
Query routes for the Ask Anything API.
"""

from fastapi import APIRouter
from pydantic import BaseModel, Field
from typing import Optional

from sqlalchemy import text

from apps.api.core.database import get_session
from apps.api.core.muhasibi_state_machine import create_middleware
from apps.api.core.schemas import FinalResponse
from apps.api.guardrails.citation_enforcer import Guardrails
from apps.api.llm.gpt5_client_azure import ProviderConfig, create_provider
from apps.api.llm.muhasibi_llm_client import MuhasibiLLMClient
from apps.api.retrieve.entity_resolver import EntityResolver
from apps.api.retrieve.hybrid_retriever import HybridRetriever

router = APIRouter()


class AskRequest(BaseModel):
    """Request model for asking a question."""

    question: str = Field(..., description="The question in Arabic or English")
    language: str = Field(default="ar", description="Preferred response language")


class Citation(BaseModel):
    """Citation reference to evidence."""

    chunk_id: str
    source_anchor: str
    ref: Optional[str] = None


class Entity(BaseModel):
    """Entity reference in the answer."""

    type: str  # pillar | core_value | sub_value
    id: str
    name_ar: str


class Purpose(BaseModel):
    """Purpose output from Muḥāsibī middleware."""

    ultimate_goal_ar: str
    constraints_ar: list[str]


class AskResponse(BaseModel):
    """
    Response model for the Ask API.

    This follows the required data contract exactly.
    """

    listen_summary_ar: str
    purpose: Purpose
    path_plan_ar: list[str]
    answer_ar: str
    citations: list[Citation]
    entities: list[Entity]
    difficulty: str  # easy | medium | hard
    not_found: bool
    confidence: str  # high | medium | low


class VectorSearchRequest(BaseModel):
    """Request for vector similarity search."""

    query: str
    top_k: int = Field(default=10, ge=1, le=100)


class GraphSearchRequest(BaseModel):
    """Request for graph traversal search."""

    entity_id: str
    depth: int = Field(default=2, ge=1, le=5)
    relationship_types: list[str] = Field(default=["CONTAINS", "SUPPORTED_BY"])


class EvidencePacket(BaseModel):
    """
    Evidence packet returned by retrieval.

    This follows the required data contract exactly.
    """

    chunk_id: str
    entity_type: str  # pillar | core_value | sub_value
    entity_id: str
    chunk_type: str  # definition | evidence | commentary
    text_ar: str
    source_doc_id: str
    source_anchor: str
    refs: list[dict]  # [{"type": "quran|hadith|book", "ref": "..."}]


class SearchResponse(BaseModel):
    """Response for search endpoints."""

    evidence_packets: list[EvidencePacket]
    total_found: int


@router.post("/ask", response_model=FinalResponse)
async def ask_question(request: AskRequest):
    """
    Ask a question about the wellbeing framework.

    This endpoint uses the Muḥāsibī reasoning middleware:
    1. LISTEN - Normalize and understand the question
    2. PURPOSE - Determine ultimate goal and constraints
    3. PATH - Plan the approach
    4. RETRIEVE - Get evidence packets
    5. ACCOUNT - Validate evidence coverage
    6. INTERPRET - Generate answer from evidence
    7. REFLECT - Add consequence-aware reflection
    8. FINALIZE - Validate and return

    If evidence is insufficient, returns not_found=true.

    Args:
        request: The question and language preference.

    Returns:
        AskResponse: Answer with citations, or refusal if no evidence.
    """
    async with get_session() as session:
        # Build resolver from DB (best-effort; if DB empty this remains empty and system will refuse)
        resolver = EntityResolver()
        try:
            pillars = (await session.execute(text("SELECT id, name_ar FROM pillar"))).fetchall()
            core_values = (await session.execute(text("SELECT id, name_ar FROM core_value"))).fetchall()
            sub_values = (await session.execute(text("SELECT id, name_ar FROM sub_value"))).fetchall()
            resolver.load_entities(
                pillars=[{"id": r.id, "name_ar": r.name_ar} for r in pillars],
                core_values=[{"id": r.id, "name_ar": r.name_ar} for r in core_values],
                sub_values=[{"id": r.id, "name_ar": r.name_ar} for r in sub_values],
            )
        except Exception:
            # DB may be unavailable in some dev contexts; proceed with empty resolver
            pass

        guardrails = Guardrails()

        retriever = HybridRetriever()
        # Attach session for middleware retrieval (keeps middleware signature stable for tests)
        retriever._session = session  # type: ignore[attr-defined]

        llm_client = None
        try:
            cfg = ProviderConfig.from_env()
            if cfg.is_configured():
                provider = create_provider(cfg)
                llm_client = MuhasibiLLMClient(provider)
        except Exception:
            llm_client = None

        middleware = create_middleware(
            entity_resolver=resolver,
            retriever=retriever,
            llm_client=llm_client,  # uses Azure/OpenAI if configured; else deterministic fallback
            guardrails=guardrails,
        )

        return await middleware.process(request.question, language=request.language)


@router.post("/search/vector", response_model=SearchResponse)
async def search_vector(request: VectorSearchRequest):
    """
    Perform vector similarity search over chunks.

    Args:
        request: Query and top-K parameter.

    Returns:
        SearchResponse: Matching evidence packets.
    """
    from apps.api.retrieve.vector_retriever import VectorRetriever

    async with get_session() as session:
        retriever = VectorRetriever()
        packets = await retriever.search(session, request.query, top_k=request.top_k)
        return SearchResponse(evidence_packets=packets, total_found=len(packets))


@router.post("/search/graph", response_model=SearchResponse)
async def search_graph(request: GraphSearchRequest):
    """
    Perform graph traversal from an entity.

    Args:
        request: Starting entity and traversal parameters.

    Returns:
        SearchResponse: Evidence packets from graph neighborhood.
    """
    from apps.api.retrieve.graph_retriever import expand_graph
    from apps.api.core.schemas import EntityType
    from apps.api.retrieve.sql_retriever import get_chunks_with_refs

    async with get_session() as session:
        # Try to infer entity type by ID prefix (P/CV/SV)
        entity_type = EntityType.SUB_VALUE
        if request.entity_id.startswith("P"):
            entity_type = EntityType.PILLAR
        elif request.entity_id.startswith("CV"):
            entity_type = EntityType.CORE_VALUE

        neighbors = await expand_graph(
            session,
            entity_type=entity_type,
            entity_id=request.entity_id,
            depth=request.depth,
            relationship_types=request.relationship_types,
        )

        packets = []
        for n in neighbors:
            n_type = n.get("neighbor_type")
            n_id = n.get("neighbor_id")
            if not n_type or not n_id:
                continue
            try:
                et = EntityType(n_type)
            except Exception:
                continue
            packets.extend(await get_chunks_with_refs(session, et, n_id, limit=10))

        return SearchResponse(evidence_packets=packets, total_found=len(packets))

