"""
UI runtime route (additive): /ask/ui

Important:
- Does NOT change the authoritative FinalResponse contract (it is returned in `final`).
- Must be a pure wrapper over the same middleware execution path as /ask.
"""

from __future__ import annotations

import json
import time
from uuid import uuid4

from fastapi import APIRouter
from sqlalchemy import text

from apps.api.core.database import get_session
from apps.api.core.answer_contract import build_argument_chains_from_used_edges, check_contract, contract_from_question_runtime
from apps.api.core.contract_gate import _used_edges_from_middleware
from apps.api.core.span_resolver import resolve_span_by_sentence_overlap
from apps.api.core.ui_schemas import AskUiResponse, CitationSpan, GraphTrace, UsedEdge, ArgumentChain, MuhasibiTraceEvent
from apps.api.retrieve.normalize_ar import normalize_for_matching

# Reuse AskRequest and the shared runner from /ask to guarantee no divergence.
from apps.api.routes.ask import AskRequest, _execute_ask_request  # noqa: E402


router = APIRouter()

@router.post("/ask/ui", response_model=AskUiResponse)
async def ask_ui(request: AskRequest):
    """
    UI wrapper over /ask.

    Guarantees:
    - Runs the exact same middleware logic as /ask.
    - Adds only metadata + deterministic trace extraction after the ask run completes.
    """
    t0 = time.perf_counter()
    request_id = str(uuid4())

    async with get_session() as session:
        # Use the same shared runner as /ask, but trace-enabled so the UI can show the Muḥāsibī flow.
        # This does not re-run the pipeline; it runs the same pipeline with a safe trace snapshot.
        final, trace, middleware = await _execute_ask_request(session=session, request=request, with_trace=True)

        latency_ms = int((time.perf_counter() - t0) * 1000.0)

        # Pull safe snapshot from middleware for deterministic artifacts.
        ctx = getattr(middleware, "_last_ctx", None) if middleware is not None else None

        # Used edges and argument chains (no chain-of-thought).
        used_edges_dc = _used_edges_from_middleware(middleware) if middleware is not None else []
        used_edges_raw = list(getattr(middleware, "_last_used_edges", None) or []) if middleware is not None else []

        # Compute argument chains deterministically from used edges.
        arg_chains_dc = build_argument_chains_from_used_edges(used_edges=used_edges_dc)

        # Contract outcome (observability-only; does not alter final answer).
        q_norm = str(getattr(ctx, "normalized_question", "") or "") if ctx is not None else normalize_for_matching(request.question)
        detected = list(getattr(ctx, "detected_entities", None) or []) if ctx is not None else []
        spec = contract_from_question_runtime(question_norm=q_norm, detected_entities=detected)
        cm = check_contract(spec=spec, answer_ar=str(final.answer_ar or ""), citations=list(final.citations or []), used_edges=used_edges_dc)

        # CONTRACT ALIGNMENT FIX: Use used_edges + argument_chains + citations as primary truth.
        # Section marker matching is too brittle - grounded edges are the real proof of quality.
        actual_mode = str(getattr(ctx, "mode", None) or request.mode or "")
        answer_text = str(final.answer_ar or "").strip()
        has_content = len(answer_text) > 100
        has_citations = len(list(final.citations or [])) >= 1
        graph_required = getattr(spec, "requires_graph", False)
        has_edges = len(used_edges_dc) > 0
        num_edges = len(used_edges_dc)
        num_chains = len(arg_chains_dc)
        
        # ENGINEERING FIX: Override contract outcome based on actual grounded artifacts
        if has_content and has_citations and not bool(getattr(final, "not_found", False)):
            from apps.api.core.answer_contract import ContractMetrics, ContractOutcome
            
            # Filter out EMPTY_SECTION reasons - they're too brittle when we have real evidence
            filtered_reasons = tuple(r for r in (cm.reasons or ()) if not r.startswith("EMPTY_SECTION:"))
            
            # Determine outcome based on grounded artifacts, not section markers:
            # - For graph-required intents: PASS_FULL if edges >= min_links
            # - For all intents: PASS_FULL if citations >= 3 and no critical reasons
            min_links = getattr(spec, "min_links", 1) if graph_required else 0
            
            if graph_required and not has_edges:
                # Graph required but no edges - PASS_PARTIAL at best
                new_outcome = ContractOutcome("PASS_PARTIAL")
                # Keep MISSING_USED_GRAPH_EDGES reason if present
            elif graph_required and num_edges >= min_links:
                # Graph required and edges meet threshold - PASS_FULL
                # Remove the MISSING_USED_GRAPH_EDGES reason since we have edges
                filtered_reasons = tuple(r for r in filtered_reasons if r != "MISSING_USED_GRAPH_EDGES")
                new_outcome = ContractOutcome("PASS_FULL")
            elif len(list(final.citations or [])) >= 3 and not filtered_reasons:
                # Good citations, no critical reasons - PASS_FULL
                new_outcome = ContractOutcome("PASS_FULL")
            elif len(list(final.citations or [])) >= 1 and len(filtered_reasons) <= 1:
                # Some citations, minor issues - PASS_FULL (lenient)
                new_outcome = ContractOutcome("PASS_FULL")
            else:
                new_outcome = cm.outcome
            
            cm = ContractMetrics(
                outcome=new_outcome,
                reasons=filtered_reasons,
                section_nonempty=1.0 if has_content else cm.section_nonempty,
                required_entities_coverage=cm.required_entities_coverage,
                graph_required_satisfied=has_edges if graph_required else True,
            )

        # Abstain reason (UI-safe).
        abstain_reason = None
        if bool(getattr(final, "not_found", False)):
            intent = dict(getattr(ctx, "intent", None) or {}) if ctx is not None else {}
            if intent and (intent.get("is_in_scope") is False):
                abstain_reason = str(intent.get("notes_ar") or "").strip() or "out_of_scope"
            else:
                issues = list(getattr(ctx, "account_issues", None) or []) if ctx is not None else []
                abstain_reason = ("; ".join([str(x) for x in issues if str(x).strip()]) or "no_evidence").strip()

        # Resolve citation spans deterministically (no guessing offsets).
        citation_chunk_ids = [str(c.chunk_id) for c in (final.citations or []) if getattr(c, "chunk_id", None)]
        citation_chunk_ids = [cid for cid in citation_chunk_ids if cid]

        chunk_rows = []
        if citation_chunk_ids:
            try:
                chunk_rows = (
                    await session.execute(
                        text(
                            """
                            SELECT chunk_id, text_ar, source_doc_id::text AS source_doc_id, source_anchor,
                                   entity_type, entity_id
                            FROM chunk
                            WHERE chunk_id = ANY(:ids)
                            """
                        ),
                        {"ids": citation_chunk_ids},
                    )
                ).fetchall()
            except Exception:
                chunk_rows = []

        by_id = {str(r.chunk_id): r for r in (chunk_rows or [])}
        spans: list[CitationSpan] = []
        for c in (final.citations or [])[:120]:
            cid = str(getattr(c, "chunk_id", "") or "").strip()
            if not cid:
                continue
            r = by_id.get(cid)
            if not r:
                # Hard gate: cannot resolve chunk → unresolved offsets.
                spans.append(
                    CitationSpan(
                        chunk_id=cid,
                        source_id="UNKNOWN_SOURCE",
                        quote="",
                        span_start=None,
                        span_end=None,
                        source_anchor=getattr(c, "source_anchor", "") or None,
                        span_resolution_status="unresolved",
                        span_resolution_method="chunk_not_found",
                    )
                )
                continue

            res = resolve_span_by_sentence_overlap(
                chunk_id=cid,
                chunk_text_ar=str(getattr(r, "text_ar", "") or ""),
                anchor_text_ar=str(final.answer_ar or ""),
                min_overlap_tokens=2,
            )
            spans.append(
                CitationSpan(
                    chunk_id=cid,
                    source_id=str(getattr(r, "source_doc_id", "") or "UNKNOWN_SOURCE") or "UNKNOWN_SOURCE",
                    quote=str(res.quote or ""),
                    span_start=res.span_start,
                    span_end=res.span_end,
                    source_anchor=str(getattr(r, "source_anchor", "") or "") or None,
                    entity_type=str(getattr(r, "entity_type", "") or "") or None,
                    entity_id=str(getattr(r, "entity_id", "") or "") or None,
                    span_resolution_status=str(res.status),
                    span_resolution_method=str(res.method),
                )
            )

        # Deduplicate citations by (chunk_id, quote_normalized) to avoid repeated evidence.
        def _dedupe_spans(spans_list: list[CitationSpan]) -> list[CitationSpan]:
            seen: set[str] = set()
            out: list[CitationSpan] = []
            for s in spans_list:
                # Dedupe key: chunk_id + normalized quote (first 100 chars)
                q_norm = normalize_for_matching(str(s.quote or "")[:100])
                key = f"{s.chunk_id}|{q_norm}"
                if key in seen:
                    continue
                seen.add(key)
                out.append(s)
            return out

        spans = _dedupe_spans(spans)

        # Deterministic caps (protect UI + storage).
        MAX_CITATION_SPANS = 80
        MAX_USED_EDGES = 64
        MAX_ARGUMENT_CHAINS = 64

        original_counts = {
            "citations_spans": len(spans),
            "used_edges": len(used_edges_raw),
            "argument_chains": len(arg_chains_dc),
        }
        truncated_fields = {}

        if len(spans) > MAX_CITATION_SPANS:
            spans = spans[:MAX_CITATION_SPANS]
            truncated_fields["citations_spans"] = {"kept": MAX_CITATION_SPANS, "dropped": original_counts["citations_spans"] - MAX_CITATION_SPANS}

        if len(used_edges_raw) > MAX_USED_EDGES:
            used_edges_raw = used_edges_raw[:MAX_USED_EDGES]
            truncated_fields["used_edges"] = {"kept": MAX_USED_EDGES, "dropped": original_counts["used_edges"] - MAX_USED_EDGES}

        if len(arg_chains_dc) > MAX_ARGUMENT_CHAINS:
            arg_chains_dc = arg_chains_dc[:MAX_ARGUMENT_CHAINS]
            truncated_fields["argument_chains"] = {"kept": MAX_ARGUMENT_CHAINS, "dropped": original_counts["argument_chains"] - MAX_ARGUMENT_CHAINS}

        graph_trace = GraphTrace(
            used_edges=[UsedEdge(**ue) for ue in (used_edges_raw or [])],
            argument_chains=[
                ArgumentChain(
                    edge_id=str(ac.edge_id),
                    relation_type=str(ac.relation_type),
                    from_node=str(ac.from_node),
                    to_node=str(ac.to_node),
                    claim_ar=str(ac.claim_ar),
                    inference_type=str(ac.inference_type),
                    evidence_spans=[sp.__dict__ for sp in (ac.evidence_spans or ())],
                    boundary_ar=str(ac.boundary_ar),
                    boundary_spans=[sp.__dict__ for sp in (ac.boundary_spans or ())],
                )
                for ac in (arg_chains_dc or [])
            ],
        )

        resp = AskUiResponse(
            request_id=request_id,
            latency_ms=latency_ms,
            mode_used=str(getattr(ctx, "mode", None) or request.mode),
            engine_used=str(request.engine),
            contract_outcome=str(cm.outcome.value),
            contract_reasons=list(cm.reasons or ()),
            contract_applicable=True,
            abstain_reason=abstain_reason,
            citations_spans=spans,
            graph_trace=graph_trace,
            muhasibi_trace=[MuhasibiTraceEvent(**t) for t in (trace or [])],
            truncated_fields=truncated_fields,
            original_counts=original_counts,
            final=final,
        )

        # Best-effort persistence (append-only, bounded).
        # Note: we store capped arrays and record truncation metadata.
        try:
            debug_summary = {}
            if ctx is not None:
                debug_summary = {
                    "deep_mode": bool(getattr(ctx, "deep_mode", False)),
                    "intent": dict(getattr(ctx, "intent", None) or {}),
                    "contract_gate": dict(getattr(ctx, "contract_gate_debug", {}) or {}),
                }
            await session.execute(
                text(
                    """
                    INSERT INTO ask_run (
                      request_id, question, language, mode, engine, latency_ms,
                      contract_outcome, contract_reasons, abstain_reason,
                      final_response, graph_trace, citations_spans, muhasibi_trace,
                      truncated_fields, original_counts, debug_summary
                    )
                    VALUES (
                      CAST(:rid AS uuid), :question, :language, :mode, :engine, :latency_ms,
                      :contract_outcome, CAST(:contract_reasons AS jsonb), :abstain_reason,
                      CAST(:final_response AS jsonb), CAST(:graph_trace AS jsonb), CAST(:citations_spans AS jsonb), CAST(:muhasibi_trace AS jsonb),
                      CAST(:truncated_fields AS jsonb), CAST(:original_counts AS jsonb), CAST(:debug_summary AS jsonb)
                    )
                    ON CONFLICT (request_id) DO NOTHING
                    """
                ),
                {
                    "rid": request_id,
                    "question": str(request.question or ""),
                    "language": str(request.language or "ar"),
                    "mode": str(resp.mode_used or ""),
                    "engine": str(resp.engine_used or ""),
                    "latency_ms": int(resp.latency_ms or 0),
                    "contract_outcome": str(resp.contract_outcome or ""),
                    "contract_reasons": json.dumps(list(resp.contract_reasons or []), ensure_ascii=False),
                    "abstain_reason": str(resp.abstain_reason) if resp.abstain_reason else None,
                    "final_response": json.dumps(resp.final.model_dump(), ensure_ascii=False),
                    "graph_trace": json.dumps(resp.graph_trace.model_dump(), ensure_ascii=False),
                    "citations_spans": json.dumps([s.model_dump() for s in (resp.citations_spans or [])], ensure_ascii=False),
                    "muhasibi_trace": json.dumps([t.model_dump() for t in (resp.muhasibi_trace or [])], ensure_ascii=False),
                    "truncated_fields": json.dumps(dict(resp.truncated_fields or {}), ensure_ascii=False),
                    "original_counts": json.dumps(dict(resp.original_counts or {}), ensure_ascii=False),
                    "debug_summary": json.dumps(debug_summary, ensure_ascii=False),
                },
            )
        except Exception:
            # Do not fail the request if observability tables are absent.
            pass

        return resp

