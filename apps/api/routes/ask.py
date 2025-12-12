"""
Query routes for the Ask Anything API.
"""

from fastapi import APIRouter
from pydantic import BaseModel, Field
from typing import Optional

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


@router.post("/ask", response_model=AskResponse)
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
    # TODO: Implement Muḥāsibī state machine in Phase 4

    # Placeholder response showing refusal behavior
    return AskResponse(
        listen_summary_ar="تم استلام السؤال. التنفيذ قيد الانتظار.",
        purpose=Purpose(
            ultimate_goal_ar="الإجابة من الأدلة فقط",
            constraints_ar=[
                "evidence_only",
                "cite_every_claim",
                "refuse_if_missing",
            ],
        ),
        path_plan_ar=["استماع", "فهم", "استرجاع", "تفسير"],
        answer_ar="لا يوجد في البيانات الحالية ما يدعم الإجابة. التنفيذ قيد الانتظار.",
        citations=[],
        entities=[],
        difficulty="medium",
        not_found=True,
        confidence="low",
    )


@router.post("/search/vector", response_model=SearchResponse)
async def search_vector(request: VectorSearchRequest):
    """
    Perform vector similarity search over chunks.

    Args:
        request: Query and top-K parameter.

    Returns:
        SearchResponse: Matching evidence packets.
    """
    # TODO: Implement in Phase 3

    return SearchResponse(evidence_packets=[], total_found=0)


@router.post("/search/graph", response_model=SearchResponse)
async def search_graph(request: GraphSearchRequest):
    """
    Perform graph traversal from an entity.

    Args:
        request: Starting entity and traversal parameters.

    Returns:
        SearchResponse: Evidence packets from graph neighborhood.
    """
    # TODO: Implement in Phase 3

    return SearchResponse(evidence_packets=[], total_found=0)

