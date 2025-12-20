"""Runner for model bakeoff: execute one model on one dataset."""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Any, Optional

from sqlalchemy import text

from apps.api.core.database import get_session
from apps.api.core.muhasibi_state_machine import create_middleware
from apps.api.guardrails.citation_enforcer import Guardrails
from apps.api.llm.gpt5_client_azure import ProviderConfig, create_provider
from apps.api.llm.muhasibi_llm_client import MuhasibiLLMClient
from apps.api.retrieve.entity_resolver import EntityResolver
from apps.api.retrieve.hybrid_retriever import HybridRetriever

from eval.datasets.io import read_dataset_jsonl
from eval.model_bakeoff_metrics import BakeoffMetrics, compute_all_metrics
from eval.types import (
    EvalCitation,
    EvalClaim,
    EvalOutputRow,
    EvalMode,
    GraphTrace,
    GraphTraceUsedEdge,
    GraphTraceUsedEdgeSpan,
    ArgumentChain,
    ArgumentInferenceType,
    ClaimSupportPolicy,
    ClaimSupportStrength,
    EvidenceSpanRef,
)


def _make_provider_for_model(model: str, temperature: float, max_tokens: int):
    """Create provider for a specific model deployment."""
    import os

    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "").strip()
    api_key = os.getenv("AZURE_OPENAI_API_KEY", "").strip()
    api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2025-01-01-preview").strip()

    cfg = ProviderConfig(
        provider_type=ProviderConfig.from_env().provider_type,
        endpoint=endpoint,
        api_key=api_key,
        api_version=api_version,
        deployment_name=model,
        model_name=model,
        max_tokens=max_tokens,
        temperature=temperature,
        timeout=120,
    )
    return create_provider(cfg)


async def _build_runtime_components(session) -> tuple[EntityResolver, Guardrails, HybridRetriever]:
    """Build runtime components for evaluation."""
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
        pass

    guardrails = Guardrails()
    retriever = HybridRetriever(enable_graph=True)
    retriever._session = session  # type: ignore
    return resolver, guardrails, retriever


def _response_to_eval_output(
    question_id: str,
    question: str,
    mode: str,
    response: Any,
    latency_ms: int,
) -> EvalOutputRow:
    """Convert middleware response to EvalOutputRow."""
    # Extract citations
    citations: list[EvalCitation] = []
    for c in getattr(response, "citations", []) or []:
        try:
            citations.append(
                EvalCitation(
                    source_id=str(getattr(c, "source_id", "") or getattr(c, "chunk_id", "") or ""),
                    span_start=int(getattr(c, "span_start", 0) or 0),
                    span_end=int(getattr(c, "span_end", 0) or 0),
                    quote=str(getattr(c, "quote", "") or "")[:200],
                )
            )
        except Exception:
            pass

    # Extract graph trace
    graph_trace = GraphTrace()
    if hasattr(response, "graph_trace") and response.graph_trace:
        gt = response.graph_trace
        graph_trace.edges = list(getattr(gt, "edges", []) or [])
        graph_trace.nodes = list(getattr(gt, "nodes", []) or [])

        # Used edges
        for ue in getattr(gt, "used_edges", []) or []:
            try:
                spans = []
                for sp in getattr(ue, "justification_spans", []) or []:
                    spans.append(
                        GraphTraceUsedEdgeSpan(
                            source_id=str(getattr(sp, "source_id", "") or getattr(sp, "chunk_id", "") or ""),
                            chunk_id=str(getattr(sp, "chunk_id", "") or getattr(sp, "source_id", "") or ""),
                            span_start=int(getattr(sp, "span_start", 0) or 0),
                            span_end=int(getattr(sp, "span_end", 0) or 0),
                            quote=str(getattr(sp, "quote", "") or "")[:200],
                        )
                    )
                graph_trace.used_edges.append(
                    GraphTraceUsedEdge(
                        edge_id=str(getattr(ue, "edge_id", "") or ""),
                        from_node=str(getattr(ue, "from_node", "") or ""),
                        to_node=str(getattr(ue, "to_node", "") or ""),
                        relation_type=str(getattr(ue, "relation_type", "") or ""),
                        justification_spans=spans,
                    )
                )
            except Exception:
                pass

        # Argument chains
        for ac in getattr(gt, "argument_chains", []) or []:
            try:
                ev_spans = []
                for sp in getattr(ac, "evidence_spans", []) or []:
                    ev_spans.append(
                        EvidenceSpanRef(
                            source_id=str(getattr(sp, "source_id", "") or ""),
                            span_start=int(getattr(sp, "span_start", 0) or 0),
                            span_end=int(getattr(sp, "span_end", 0) or 0),
                            quote=str(getattr(sp, "quote", "") or "")[:200],
                        )
                    )
                graph_trace.argument_chains.append(
                    ArgumentChain(
                        edge_id=str(getattr(ac, "edge_id", "") or ""),
                        relation_type=str(getattr(ac, "relation_type", "") or ""),
                        from_node=str(getattr(ac, "from_node", "") or ""),
                        to_node=str(getattr(ac, "to_node", "") or ""),
                        claim_ar=str(getattr(ac, "claim_ar", "") or ""),
                        inference_type=ArgumentInferenceType.DIRECT_QUOTE,
                        evidence_spans=ev_spans,
                        boundary_ar=str(getattr(ac, "boundary_ar", "") or ""),
                    )
                )
            except Exception:
                pass

    # Extract claims (simplified)
    claims: list[EvalClaim] = []
    # We'll mark all non-abstained answers as having claims requiring evidence
    if not bool(getattr(response, "not_found", False)):
        claims.append(
            EvalClaim(
                claim_id="main",
                text_ar=str(getattr(response, "answer_ar", "") or "")[:500],
                support_strength=ClaimSupportStrength.DIRECT if citations else ClaimSupportStrength.NONE_ALLOWED,
                support_policy=ClaimSupportPolicy.MUST_CITE,
                requires_evidence=True,
            )
        )

    return EvalOutputRow(
        id=question_id,
        mode=EvalMode.FULL_SYSTEM,
        question=question,
        answer_ar=str(getattr(response, "answer_ar", "") or ""),
        claims=claims,
        citations=citations,
        graph_trace=graph_trace,
        abstained=bool(getattr(response, "not_found", False)),
        latency_ms=latency_ms,
    )


async def run_model_on_dataset(
    *,
    model: str,
    dataset_path: Path,
    mode: str,
    seed: int,
    temperature: float,
    max_tokens: int,
    top_k: int,
    timeout: int,
) -> BakeoffMetrics:
    """Run one model on one dataset and compute metrics."""
    rows = read_dataset_jsonl(dataset_path)
    dataset_by_id = {r.id: r.model_dump() for r in rows}

    outputs: list[EvalOutputRow] = []
    integrity_hits = 0

    async with get_session() as session:
        resolver, guardrails, retriever = await _build_runtime_components(session)
        provider = _make_provider_for_model(model, temperature, max_tokens)
        llm_client = MuhasibiLLMClient(provider)

        middleware = create_middleware(
            entity_resolver=resolver,
            retriever=retriever,
            llm_client=llm_client,
            guardrails=guardrails,
        )

        for r in rows:
            t0 = time.perf_counter()
            try:
                response = await asyncio.wait_for(
                    middleware.process(r.question_ar, language="ar", mode=mode),
                    timeout=timeout,
                )
                latency_ms = int((time.perf_counter() - t0) * 1000)

                output = _response_to_eval_output(
                    question_id=r.id,
                    question=r.question_ar,
                    mode=mode,
                    response=response,
                    latency_ms=latency_ms,
                )
                outputs.append(output)

            except asyncio.TimeoutError:
                latency_ms = int((time.perf_counter() - t0) * 1000)
                outputs.append(
                    EvalOutputRow(
                        id=r.id,
                        mode=EvalMode.FULL_SYSTEM,
                        question=r.question_ar,
                        answer_ar="",
                        abstained=True,
                        abstain_reason="timeout",
                        latency_ms=latency_ms,
                    )
                )
            except Exception as e:
                latency_ms = int((time.perf_counter() - t0) * 1000)
                outputs.append(
                    EvalOutputRow(
                        id=r.id,
                        mode=EvalMode.FULL_SYSTEM,
                        question=r.question_ar,
                        answer_ar="",
                        abstained=True,
                        abstain_reason=f"error: {str(e)[:50]}",
                        latency_ms=latency_ms,
                    )
                )

    return compute_all_metrics(outputs, dataset_by_id, integrity_hits=integrity_hits)
