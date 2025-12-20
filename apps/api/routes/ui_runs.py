"""
UI run storage + replay + feedback + chunk endpoints (additive).

Reason:
- Keep files under 500 LOC.
- Separate runtime /ask/ui execution from replay/feedback helpers.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from sqlalchemy import text

from apps.api.core.database import get_session
from apps.api.core.ui_schemas import AskUiResponse

router = APIRouter()


class ChunkRef(BaseModel):
    ref_type: str
    ref: str


class ChunkResponse(BaseModel):
    chunk_id: str
    entity_type: str
    entity_id: str
    chunk_type: str
    text_ar: str
    text_en: str | None = None
    source_doc_id: str | None = None
    source_anchor: str | None = None
    refs: list[ChunkRef] = []


@router.get("/chunk/{chunk_id}", response_model=ChunkResponse)
async def get_chunk(chunk_id: str):
    """Fetch a stored chunk by id (UI 'open chunk')."""
    cid = str(chunk_id or "").strip()
    if not cid:
        raise HTTPException(status_code=400, detail="chunk_id is required")

    async with get_session() as session:
        row = (
            await session.execute(
                text(
                    """
                    SELECT chunk_id, entity_type, entity_id, chunk_type,
                           text_ar, text_en,
                           source_doc_id::text AS source_doc_id,
                           source_anchor
                    FROM chunk
                    WHERE chunk_id = :cid
                    """
                ),
                {"cid": cid},
            )
        ).fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="chunk not found")

        ref_rows = (
            await session.execute(
                text(
                    """
                    SELECT ref_type, ref
                    FROM chunk_ref
                    WHERE chunk_id = :cid
                    ORDER BY ref_type, ref
                    """
                ),
                {"cid": cid},
            )
        ).fetchall()

        return ChunkResponse(
            chunk_id=str(row.chunk_id),
            entity_type=str(row.entity_type),
            entity_id=str(row.entity_id),
            chunk_type=str(row.chunk_type),
            text_ar=str(row.text_ar or ""),
            text_en=str(row.text_en) if getattr(row, "text_en", None) is not None else None,
            source_doc_id=str(row.source_doc_id) if getattr(row, "source_doc_id", None) else None,
            source_anchor=str(row.source_anchor) if getattr(row, "source_anchor", None) else None,
            refs=[ChunkRef(ref_type=str(r.ref_type), ref=str(r.ref)) for r in (ref_rows or [])],
        )


class FeedbackRequest(BaseModel):
    request_id: str | None = None
    rating: int | None = None
    tags: list[str] = []
    comment: str | None = None


class FeedbackResponse(BaseModel):
    stored: bool


@router.post("/feedback", response_model=FeedbackResponse)
async def submit_feedback(req: FeedbackRequest):
    """Store feedback linked to an ask run (append-only)."""
    rid = str(req.request_id or "").strip()
    if not rid:
        raise HTTPException(status_code=400, detail="request_id is required")

    async with get_session() as session:
        try:
            await session.execute(
                text(
                    """
                    INSERT INTO ask_feedback (request_id, rating, tags, comment)
                    VALUES (CAST(:rid AS uuid), :rating, :tags, :comment)
                    """
                ),
                {
                    "rid": rid,
                    "rating": int(req.rating) if req.rating is not None else None,
                    "tags": list(req.tags or [])[:12],
                    "comment": str(req.comment)[:4000] if req.comment else None,
                },
            )
            return FeedbackResponse(stored=True)
        except Exception:
            return FeedbackResponse(stored=False)


class AskRunBundleResponse(BaseModel):
    ask: AskUiResponse
    debug_summary: dict = {}


@router.get("/ask/runs/{request_id}", response_model=AskUiResponse)
async def get_ask_run(request_id: str):
    """Replay a stored ask run (requires ask_run table)."""
    rid = str(request_id or "").strip()
    if not rid:
        raise HTTPException(status_code=400, detail="request_id is required")

    async with get_session() as session:
        row = (
            await session.execute(
                text(
                    """
                    SELECT
                      request_id::text AS request_id,
                      latency_ms,
                      mode,
                      engine,
                      contract_outcome,
                      contract_reasons,
                      abstain_reason,
                      final_response,
                      graph_trace,
                      citations_spans,
                      muhasibi_trace,
                      truncated_fields,
                      original_counts
                    FROM ask_run
                    WHERE request_id = CAST(:rid AS uuid)
                    """
                ),
                {"rid": rid},
            )
        ).fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="ask run not found")

        payload = {
            "request_id": str(row.request_id),
            "latency_ms": int(row.latency_ms or 0),
            "mode_used": str(row.mode or ""),
            "engine_used": str(row.engine or ""),
            "contract_outcome": str(row.contract_outcome or "FAIL"),
            "contract_reasons": list(row.contract_reasons or []),
            "contract_applicable": True,
            "abstain_reason": str(row.abstain_reason) if row.abstain_reason else None,
            "citations_spans": list(row.citations_spans or []),
            "graph_trace": dict(row.graph_trace or {}),
            "muhasibi_trace": list(row.muhasibi_trace or []),
            "truncated_fields": dict(row.truncated_fields or {}),
            "original_counts": dict(row.original_counts or {}),
            "final": dict(row.final_response or {}),
        }
        return AskUiResponse(**payload)


@router.get("/ask/runs/{request_id}/bundle", response_model=AskRunBundleResponse)
async def get_ask_run_bundle(request_id: str):
    """Support bundle: stored AskUiResponse plus a small server-side debug summary."""
    rid = str(request_id or "").strip()
    if not rid:
        raise HTTPException(status_code=400, detail="request_id is required")

    async with get_session() as session:
        row = (
            await session.execute(
                text(
                    """
                    SELECT
                      request_id::text AS request_id,
                      latency_ms,
                      mode,
                      engine,
                      contract_outcome,
                      contract_reasons,
                      abstain_reason,
                      final_response,
                      graph_trace,
                      citations_spans,
                      muhasibi_trace,
                      truncated_fields,
                      original_counts,
                      debug_summary
                    FROM ask_run
                    WHERE request_id = CAST(:rid AS uuid)
                    """
                ),
                {"rid": rid},
            )
        ).fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="ask run not found")

        payload = {
            "request_id": str(row.request_id),
            "latency_ms": int(row.latency_ms or 0),
            "mode_used": str(row.mode or ""),
            "engine_used": str(row.engine or ""),
            "contract_outcome": str(row.contract_outcome or "FAIL"),
            "contract_reasons": list(row.contract_reasons or []),
            "contract_applicable": True,
            "abstain_reason": str(row.abstain_reason) if row.abstain_reason else None,
            "citations_spans": list(row.citations_spans or []),
            "graph_trace": dict(row.graph_trace or {}),
            "muhasibi_trace": list(row.muhasibi_trace or []),
            "truncated_fields": dict(row.truncated_fields or {}),
            "original_counts": dict(row.original_counts or {}),
            "final": dict(row.final_response or {}),
        }
        return AskRunBundleResponse(ask=AskUiResponse(**payload), debug_summary=dict(row.debug_summary or {}))

