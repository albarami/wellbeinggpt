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

async def _build_runtime_components(session) -> tuple[EntityResolver, Guardrails, HybridRetriever]:
    """
    Build resolver/guardrails/retriever bound to the current DB session.

    Reason: /ask, /ask/trace, and /ask/ui must share the exact same runtime setup.
    """
    resolver = EntityResolver()
    try:
        pillars = (await session.execute(text("SELECT id, name_ar FROM pillar"))).fetchall()
        core_values = (await session.execute(text("SELECT id, name_ar FROM core_value"))).fetchall()
        sub_values = (await session.execute(text("SELECT id, name_ar FROM sub_value"))).fetchall()
        resolver.load_entities(
            pillars=[{"id": str(r.id), "name_ar": r.name_ar} for r in pillars],
            core_values=[{"id": str(r.id), "name_ar": r.name_ar} for r in core_values],
            sub_values=[{"id": str(r.id), "name_ar": r.name_ar} for r in sub_values],
            aliases_path="data/static/aliases_ar.json",
        )
    except Exception:
        # DB may be unavailable in some dev contexts; proceed with empty resolver.
        pass

    guardrails = Guardrails()

    retriever = HybridRetriever()
    # Attach session for middleware retrieval (keeps middleware signature stable for tests)
    retriever._session = session  # type: ignore[attr-defined]

    return resolver, guardrails, retriever


def _build_llm_client_or_none(model_deployment: Optional[str] = None) -> Optional[MuhasibiLLMClient]:
    """
    Build the LLM client if configured, otherwise None.

    Args:
        model_deployment: Optional deployment name override (gpt-5-chat, gpt-5.1, gpt-5.2)

    Reason: keep runtime behavior identical across /ask and /ask/ui.
    """
    llm_client = None
    try:
        cfg = ProviderConfig.from_env()
        # Override deployment if specified
        if model_deployment:
            cfg = ProviderConfig(
                provider_type=cfg.provider_type,
                endpoint=cfg.endpoint,
                api_key=cfg.api_key,
                api_version=cfg.api_version,
                deployment_name=model_deployment,  # Use the override
                model_name=cfg.model_name,
                max_tokens=cfg.max_tokens,
                temperature=cfg.temperature,
                timeout=cfg.timeout,
            )
        import logging

        logging.getLogger(__name__).info(
            f"[ASK] LLM configured: {cfg.is_configured()}, deployment: {cfg.deployment_name}, endpoint: {cfg.endpoint[:50] if cfg.endpoint else 'None'}"
        )
        if cfg.is_configured():
            provider = create_provider(cfg)
            llm_client = MuhasibiLLMClient(provider)
    except Exception as e:
        import logging

        logging.getLogger(__name__).error(f"[ASK] LLM setup failed: {e}")
        llm_client = None
    return llm_client


async def _execute_ask_request(
    *,
    session,
    request: "AskRequest",
    with_trace: bool,
) -> tuple[FinalResponse, list[dict], Optional[object]]:
    """
    Execute the ask pipeline using the shared runtime components.

    Contract:
    - This is the single shared code path used by /ask, /ask/trace, and /ask/ui.
    - /ask/ui is allowed to add metadata extraction and persistence AFTER this call.
    """
    from apps.api.core.baseline_answer import generate_baseline_answer

    resolver, guardrails, retriever = await _build_runtime_components(session)

    # Use baseline mode if requested
    if request.engine == "baseline":
        final = await generate_baseline_answer(
            question=request.question,
            retriever=retriever,
            resolver=resolver,
            guardrails=guardrails,
            language=request.language,
        )
        final = _fail_closed_if_invalid(final)
        return final, [], None

    llm_client = _build_llm_client_or_none(model_deployment=request.model_deployment)
    middleware = create_middleware(
        entity_resolver=resolver,
        retriever=retriever,
        llm_client=llm_client,  # uses Azure/OpenAI if configured; else deterministic fallback
        guardrails=guardrails,
    )

    if with_trace:
        final, trace = await middleware.process_with_trace(
            request.question, language=request.language, mode=request.mode
        )
        final = _fail_closed_if_invalid(final)
        return final, trace, middleware

    final = await middleware.process(request.question, language=request.language, mode=request.mode)

    import logging

    logger = logging.getLogger(__name__)
    logger.info(
        f"[ASK] Before fail_closed: not_found={final.not_found}, citations={len(final.citations)}, answer_len={len(final.answer_ar)}"
    )

    final = _fail_closed_if_invalid(final)
    return final, [], middleware


def _fail_closed_if_invalid(final: FinalResponse) -> FinalResponse:
    """
    Fail closed for enterprise safety.

    Reason: If LLM misbehaves (no citations / empty answer) we must refuse.
    This is a last-resort safety net in addition to guardrails.
    """
    try:
        if (not final.not_found) and (not getattr(final, "citations", None) or len(final.citations) == 0):
            return FinalResponse(
                listen_summary_ar=final.listen_summary_ar,
                purpose=final.purpose,
                path_plan_ar=final.path_plan_ar,
                answer_ar="لا يوجد في البيانات الحالية ما يدعم الإجابة على هذا السؤال.",
                citations=[],
                entities=[],
                difficulty=final.difficulty,
                not_found=True,
                confidence="low",  # type: ignore[arg-type]
            )
        if (not final.not_found) and (not (final.answer_ar or "").strip()):
            return FinalResponse(
                listen_summary_ar=final.listen_summary_ar,
                purpose=final.purpose,
                path_plan_ar=final.path_plan_ar,
                answer_ar="لا يوجد في البيانات الحالية ما يدعم الإجابة على هذا السؤال.",
                citations=[],
                entities=[],
                difficulty=final.difficulty,
                not_found=True,
                confidence="low",  # type: ignore[arg-type]
            )
    except Exception:
        pass
    return final


class AskRequest(BaseModel):
    """Request model for asking a question."""

    question: str = Field(..., description="The question in Arabic or English")
    language: str = Field(default="ar", description="Preferred response language")
    mode: str = Field(
        default="answer",
        description="answer|debate|socratic|judge (controls style; still evidence-only)",
    )
    engine: str = Field(
        default="muhasibi",
        description="muhasibi|baseline - reasoning engine to use",
    )
    model_deployment: Optional[str] = Field(
        default=None,
        description="Override Azure deployment name (gpt-5-chat, gpt-5.1, gpt-5.2)",
    )
    reranker_enabled: Optional[bool] = Field(
        default=None,
        description="Override reranker setting (None=use env default, True=force on, False=force off)",
    )


class TraceEvent(BaseModel):
    """Safe Muḥāsibī trace event (no chain-of-thought)."""

    state: str
    elapsed_s: float
    mode: str
    language: str
    # optional state metadata
    detected_entities_count: Optional[int] = None
    keywords_count: Optional[int] = None
    listen_summary_ar: Optional[str] = None
    ultimate_goal_ar: Optional[str] = None
    constraints_count: Optional[int] = None
    path_steps: Optional[list[str]] = None
    evidence_packets_count: Optional[int] = None
    has_definition: Optional[bool] = None
    has_evidence: Optional[bool] = None
    not_found: Optional[bool] = None
    issues: Optional[list[str]] = None
    confidence: Optional[str] = None
    citations_count: Optional[int] = None
    reflection_added: Optional[bool] = None
    done: Optional[bool] = None


class AskTraceResponse(BaseModel):
    final_response: FinalResponse
    trace: list[TraceEvent]


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
            - engine: "muhasibi" (default) for full reasoning, "baseline" for simple evidence retrieval

    Returns:
        AskResponse: Answer with citations, or refusal if no evidence.
    """
    async with get_session() as session:
        final, _, _ = await _execute_ask_request(session=session, request=request, with_trace=False)
        return final


@router.post("/ask/trace", response_model=AskTraceResponse)
async def ask_question_with_trace(request: AskRequest):
    """
    Ask a question and return a safe Muḥāsibī trace (state flow + timings).
    """
    async with get_session() as session:
        final, trace, _ = await _execute_ask_request(session=session, request=request, with_trace=True)
        return AskTraceResponse(final_response=final, trace=[TraceEvent(**t) for t in trace])


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

