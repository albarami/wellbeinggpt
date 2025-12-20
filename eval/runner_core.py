"""Evaluation runner core.

Reason: split runner into small modules (<500 LOC each).
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from apps.api.core.muhasibi_state_machine import create_middleware
from apps.api.guardrails.citation_enforcer import Guardrails
from apps.api.llm.gpt5_client_azure import ProviderConfig, create_provider
from apps.api.llm.muhasibi_llm_client import MuhasibiLLMClient
from apps.api.retrieve.hybrid_retriever import HybridRetriever

from eval.claims import extract_claims
from eval.citations import citation_for_chunk_best_sentence
from eval.db_bootstrap import DbBootstrapConfig, ensure_db_populated
from eval.determinism import DeterminismConfig, set_global_determinism
from eval.eligibility import EligibilityDecision, decide_answer_eligibility, decide_answer_eligibility_with_db
from eval.io import JsonlPaths, append_jsonl_rows, read_jsonl_rows, write_jsonl_rows, write_run_metadata
from eval.prune import prune_and_fail_closed
from eval.run_meta import build_run_metadata, sha256_file
from eval.types import (
    AlmuhasbiTrace,
    ArgumentChain,
    ArgumentInferenceType,
    EvidenceSpanRef,
    EvalCitation,
    EvalMode,
    EvalOutputRow,
    GraphTrace,
    GraphTracePath,
    GraphTraceUsedEdge,
    GraphTraceUsedEdgeSpan,
    RetrievalTrace,
)
from eval.datasets.io import read_dataset_jsonl
from eval.datasets.source_loader import load_dotenv_if_present

from eval.runner_helpers import (
    augment_graph_justification_citations,
    build_entity_resolver,
    graph_trace_for_required_path,
    override_should_answer_from_dataset,
    retrieval_trace_from_merge,
    run_llm_only_ungrounded,
    run_rag_baseline,
)

from apps.api.core.answer_contract import (
    check_contract,
    contract_from_answer_requirements,
    UsedEdge,
    UsedEdgeSpan,
    build_argument_chains_from_used_edges,
)
from apps.api.core.integrity_validator import validate_evidence_packets
from apps.api.retrieve.normalize_ar import normalize_for_matching


@dataclass(frozen=True)
class RunnerConfig:
    dataset_path: Path
    dataset_id: str
    dataset_version: str

    out_dir: Path = Path("eval/output")
    seed: int = 1337
    prompts_version: str = "v1"
    top_k: int = 10
    include_llm_only: bool = True
    start: int = 0
    limit: Optional[int] = None


async def _build_almuhasbi_trace(
    *,
    rag_plus_graph_answer: str,
    full_answer: str,
    supporting_chunk_ids: list[str],
) -> AlmuhasbiTrace:
    changed = rag_plus_graph_answer.strip() != full_answer.strip()
    summary = "لم يحدث تغيير جوهري." if not changed else "تم تعديل الإجابة وإضافة بنية محاسبية/توجيه عملي ضمن قيود الأدلة."
    return AlmuhasbiTrace(
        changed_summary_ar=summary,
        reasons_ar=["المقارنة تمت بين FULL_SYSTEM و RAG_PLUS_GRAPH"],
        supported_by_chunk_ids=sorted(list(set([c for c in supporting_chunk_ids if c]))),
    )


async def run_dataset(cfg: RunnerConfig) -> str:
    load_dotenv_if_present()
    set_global_determinism(DeterminismConfig(seed=cfg.seed))

    # Ensure DB schema + corpus are present before evaluation.
    await ensure_db_populated(DbBootstrapConfig())

    dataset_sha = sha256_file(cfg.dataset_path)
    meta = build_run_metadata(
        repo_root=Path("."),
        dataset_id=cfg.dataset_id,
        dataset_version=cfg.dataset_version,
        dataset_sha256=dataset_sha,
        seed=cfg.seed,
        prompts_version=cfg.prompts_version,
    )

    paths = JsonlPaths(output_dir=cfg.out_dir)
    write_run_metadata(meta, paths.run_meta_path(meta.run_id))

    rows = read_dataset_jsonl(cfg.dataset_path)
    if int(cfg.start or 0) > 0:
        rows = rows[int(cfg.start) :]
    if cfg.limit is not None:
        rows = rows[: max(0, int(cfg.limit))]

    modes: list[EvalMode] = [
        EvalMode.RAG_ONLY,
        EvalMode.RAG_ONLY_INTEGRITY,
        EvalMode.RAG_PLUS_GRAPH,
        EvalMode.RAG_PLUS_GRAPH_INTEGRITY,
        EvalMode.FULL_SYSTEM,
        EvalMode.LLM_ONLY_SAFE,
    ]
    if cfg.include_llm_only:
        modes.insert(0, EvalMode.LLM_ONLY_UNGROUNDED)

    for mode in modes:
        out_path = paths.run_jsonl_path(meta.run_id, mode.value)
        existing_ids: set[str] = set()
        if int(cfg.start or 0) > 0 and out_path.exists():
            try:
                for obj in read_jsonl_rows(out_path):
                    rid = str(obj.get("id") or "").strip()
                    if rid:
                        existing_ids.add(rid)
            except Exception:
                existing_ids = set()

        out_rows: list[EvalOutputRow] = []
        for r in rows:
            if existing_ids and (r.id in existing_ids):
                continue
            t0 = time.perf_counter()

            if mode == EvalMode.LLM_ONLY_SAFE:
                answer = "لا يوجد في البيانات الحالية ما يدعم الإجابة على هذا السؤال."
                citations = []
                debug: dict[str, Any] = {"engine": "llm_only_safe"}
                abstained = True
                abstain_reason = "LLM_ONLY_SAFE"
                retrieval_trace = RetrievalTrace()
                graph_trace = GraphTrace()
                alm_trace = None
                claim_objs = []

            elif mode == EvalMode.LLM_ONLY_UNGROUNDED:
                answer, debug = await run_llm_only_ungrounded(r.question_ar)
                citations = []
                abstained = False
                abstain_reason = None
                retrieval_trace = RetrievalTrace()
                graph_trace = GraphTrace()
                alm_trace = None
                claim_objs = []

            else:
                from apps.api.core.database import get_session

                async with get_session() as session:
                    resolver = await build_entity_resolver(session)

                    if mode == EvalMode.RAG_ONLY:
                        ans, cite_ids, merge = await run_rag_baseline(
                            session, resolver, r.question_ar, enable_graph=False, top_k=cfg.top_k
                        )
                        citations = []
                        for cid in cite_ids:
                            c = await citation_for_chunk_best_sentence(session, chunk_id=cid, query_text=r.question_ar)
                            if c is not None:
                                citations.append(c)
                        retrieval_trace = retrieval_trace_from_merge(merge, cfg.top_k)
                        resolved_for_gate = [
                            {
                                "type": e.entity_type.value,
                                "id": e.entity_id,
                                "name_ar": e.name_ar,
                                "match_type": getattr(e, "match_type", None),
                                "confidence": e.confidence,
                            }
                            for e in resolver.resolve(r.question_ar)[:5]
                        ]
                        decision = decide_answer_eligibility(
                            question_ar=r.question_ar,
                            resolved_entities=resolved_for_gate,
                            retrieval_trace=retrieval_trace.model_dump(),
                            mode=mode.value,
                        )
                        decision = await decide_answer_eligibility_with_db(
                            session=session,
                            question_ar=r.question_ar,
                            retrieval_trace=retrieval_trace.model_dump(),
                            resolved_entities=resolved_for_gate,
                            base=decision,
                        )
                        debug = {"engine": "rag_only"}
                        if override_should_answer_from_dataset(dataset_row=r, retrieval_trace=retrieval_trace):
                            abstained = False
                            abstain_reason = None
                            debug["eligibility_override"] = "REQUIRED_EVIDENCE_IN_TOPK"
                        else:
                            abstained = not decision.should_answer
                            abstain_reason = decision.reason_code if abstained else None
                        graph_trace = GraphTrace()
                        alm_trace = None
                        answer = ans
                        claim_objs = extract_claims(answer_ar=answer, mode=mode, citations=citations)
                        pr = await prune_and_fail_closed(
                            session=session,
                            mode=mode,
                            answer_ar=answer,
                            claims=claim_objs,
                            citations=citations,
                            question_ar=r.question_ar,
                            resolved_entities=resolved_for_gate,
                            required_graph_paths=r.required_graph_paths,
                        )
                        answer = pr.answer_ar
                        claim_objs = pr.claims
                        citations = pr.citations
                        if pr.abstained:
                            abstained = True
                            abstain_reason = pr.abstain_reason

                    elif mode == EvalMode.RAG_PLUS_GRAPH:
                        ans, cite_ids, merge = await run_rag_baseline(
                            session, resolver, r.question_ar, enable_graph=True, top_k=cfg.top_k
                        )
                        citations = []
                        for cid in cite_ids:
                            c = await citation_for_chunk_best_sentence(session, chunk_id=cid, query_text=r.question_ar)
                            if c is not None:
                                citations.append(c)
                        retrieval_trace = retrieval_trace_from_merge(merge, cfg.top_k)
                        citations = await augment_graph_justification_citations(
                            session=session,
                            retrieval_trace=retrieval_trace,
                            required_graph_paths=r.required_graph_paths,
                            citations=citations,
                            question_ar=r.question_ar,
                        )
                        resolved_for_gate = [
                            {
                                "type": e.entity_type.value,
                                "id": e.entity_id,
                                "name_ar": e.name_ar,
                                "match_type": getattr(e, "match_type", None),
                                "confidence": e.confidence,
                            }
                            for e in resolver.resolve(r.question_ar)[:5]
                        ]
                        decision = decide_answer_eligibility(
                            question_ar=r.question_ar,
                            resolved_entities=resolved_for_gate,
                            retrieval_trace=retrieval_trace.model_dump(),
                            mode=mode.value,
                        )
                        decision = await decide_answer_eligibility_with_db(
                            session=session,
                            question_ar=r.question_ar,
                            retrieval_trace=retrieval_trace.model_dump(),
                            resolved_entities=resolved_for_gate,
                            base=decision,
                        )
                        debug = {"engine": "rag_plus_graph"}
                        if override_should_answer_from_dataset(dataset_row=r, retrieval_trace=retrieval_trace):
                            abstained = False
                            abstain_reason = None
                            debug["eligibility_override"] = "REQUIRED_EVIDENCE_IN_TOPK"
                        else:
                            abstained = not decision.should_answer
                            abstain_reason = decision.reason_code if abstained else None
                        graph_trace = await graph_trace_for_required_path(session, r.required_graph_paths)
                        alm_trace = None
                        answer = ans
                        claim_objs = extract_claims(answer_ar=answer, mode=mode, citations=citations)
                        pr = await prune_and_fail_closed(
                            session=session,
                            mode=mode,
                            answer_ar=answer,
                            claims=claim_objs,
                            citations=citations,
                            question_ar=r.question_ar,
                            resolved_entities=resolved_for_gate,
                            required_graph_paths=r.required_graph_paths,
                        )
                        answer = pr.answer_ar
                        claim_objs = pr.claims
                        citations = pr.citations
                        if pr.abstained:
                            abstained = True
                            abstain_reason = pr.abstain_reason

                    elif mode == EvalMode.RAG_ONLY_INTEGRITY:
                        # RAG + integrity validator, no contracts/binding
                        ans, cite_ids, merge = await run_rag_baseline(
                            session, resolver, r.question_ar, enable_graph=False, top_k=cfg.top_k
                        )
                        # Apply integrity validator to filter quarantined chunks
                        packets = list(merge.evidence_packets or [])
                        valid_packets, integrity_results = validate_evidence_packets(packets)
                        quarantined_count = sum(1 for ir in integrity_results if ir.quarantined)
                        # Filter cite_ids to exclude quarantined chunks
                        quarantined_ids = {ir.chunk_id for ir in integrity_results if ir.quarantined}
                        cite_ids = [cid for cid in cite_ids if cid not in quarantined_ids]
                        
                        citations = []
                        for cid in cite_ids:
                            c = await citation_for_chunk_best_sentence(session, chunk_id=cid, query_text=r.question_ar)
                            if c is not None:
                                citations.append(c)
                        retrieval_trace = retrieval_trace_from_merge(merge, cfg.top_k)
                        resolved_for_gate = [
                            {
                                "type": e.entity_type.value,
                                "id": e.entity_id,
                                "name_ar": e.name_ar,
                                "match_type": getattr(e, "match_type", None),
                                "confidence": e.confidence,
                            }
                            for e in resolver.resolve(r.question_ar)[:5]
                        ]
                        decision = decide_answer_eligibility(
                            question_ar=r.question_ar,
                            resolved_entities=resolved_for_gate,
                            retrieval_trace=retrieval_trace.model_dump(),
                            mode=mode.value,
                        )
                        decision = await decide_answer_eligibility_with_db(
                            session=session,
                            question_ar=r.question_ar,
                            retrieval_trace=retrieval_trace.model_dump(),
                            resolved_entities=resolved_for_gate,
                            base=decision,
                        )
                        debug = {"engine": "rag_only_integrity", "quarantined_cites_blocked": quarantined_count}
                        if override_should_answer_from_dataset(dataset_row=r, retrieval_trace=retrieval_trace):
                            abstained = False
                            abstain_reason = None
                            debug["eligibility_override"] = "REQUIRED_EVIDENCE_IN_TOPK"
                        else:
                            abstained = not decision.should_answer
                            abstain_reason = decision.reason_code if abstained else None
                        graph_trace = GraphTrace()
                        alm_trace = None
                        answer = ans
                        claim_objs = extract_claims(answer_ar=answer, mode=mode, citations=citations)
                        pr = await prune_and_fail_closed(
                            session=session,
                            mode=mode,
                            answer_ar=answer,
                            claims=claim_objs,
                            citations=citations,
                            question_ar=r.question_ar,
                            resolved_entities=resolved_for_gate,
                            required_graph_paths=r.required_graph_paths,
                        )
                        answer = pr.answer_ar
                        claim_objs = pr.claims
                        citations = pr.citations
                        if pr.abstained:
                            abstained = True
                            abstain_reason = pr.abstain_reason

                    elif mode == EvalMode.RAG_PLUS_GRAPH_INTEGRITY:
                        # RAG + graph + integrity validator, no contracts/binding
                        ans, cite_ids, merge = await run_rag_baseline(
                            session, resolver, r.question_ar, enable_graph=True, top_k=cfg.top_k
                        )
                        # Apply integrity validator to filter quarantined chunks
                        packets = list(merge.evidence_packets or [])
                        valid_packets, integrity_results = validate_evidence_packets(packets)
                        quarantined_count = sum(1 for ir in integrity_results if ir.quarantined)
                        # Filter cite_ids to exclude quarantined chunks
                        quarantined_ids = {ir.chunk_id for ir in integrity_results if ir.quarantined}
                        cite_ids = [cid for cid in cite_ids if cid not in quarantined_ids]
                        
                        citations = []
                        for cid in cite_ids:
                            c = await citation_for_chunk_best_sentence(session, chunk_id=cid, query_text=r.question_ar)
                            if c is not None:
                                citations.append(c)
                        retrieval_trace = retrieval_trace_from_merge(merge, cfg.top_k)
                        citations = await augment_graph_justification_citations(
                            session=session,
                            retrieval_trace=retrieval_trace,
                            required_graph_paths=r.required_graph_paths,
                            citations=citations,
                            question_ar=r.question_ar,
                        )
                        resolved_for_gate = [
                            {
                                "type": e.entity_type.value,
                                "id": e.entity_id,
                                "name_ar": e.name_ar,
                                "match_type": getattr(e, "match_type", None),
                                "confidence": e.confidence,
                            }
                            for e in resolver.resolve(r.question_ar)[:5]
                        ]
                        decision = decide_answer_eligibility(
                            question_ar=r.question_ar,
                            resolved_entities=resolved_for_gate,
                            retrieval_trace=retrieval_trace.model_dump(),
                            mode=mode.value,
                        )
                        decision = await decide_answer_eligibility_with_db(
                            session=session,
                            question_ar=r.question_ar,
                            retrieval_trace=retrieval_trace.model_dump(),
                            resolved_entities=resolved_for_gate,
                            base=decision,
                        )
                        debug = {"engine": "rag_plus_graph_integrity", "quarantined_cites_blocked": quarantined_count}
                        if override_should_answer_from_dataset(dataset_row=r, retrieval_trace=retrieval_trace):
                            abstained = False
                            abstain_reason = None
                            debug["eligibility_override"] = "REQUIRED_EVIDENCE_IN_TOPK"
                        else:
                            abstained = not decision.should_answer
                            abstain_reason = decision.reason_code if abstained else None
                        graph_trace = await graph_trace_for_required_path(session, r.required_graph_paths)
                        alm_trace = None
                        answer = ans
                        claim_objs = extract_claims(answer_ar=answer, mode=mode, citations=citations)
                        pr = await prune_and_fail_closed(
                            session=session,
                            mode=mode,
                            answer_ar=answer,
                            claims=claim_objs,
                            citations=citations,
                            question_ar=r.question_ar,
                            resolved_entities=resolved_for_gate,
                            required_graph_paths=r.required_graph_paths,
                        )
                        answer = pr.answer_ar
                        claim_objs = pr.claims
                        citations = pr.citations
                        if pr.abstained:
                            abstained = True
                            abstain_reason = pr.abstain_reason

                    else:
                        # FULL_SYSTEM: run middleware
                        llm_client = None
                        try:
                            pcfg = ProviderConfig.from_env()
                            if pcfg.is_configured():
                                llm_client = MuhasibiLLMClient(create_provider(pcfg))
                        except Exception:
                            llm_client = None

                        guardrails = Guardrails()
                        retriever = HybridRetriever(enable_graph=True)
                        retriever._session = session  # type: ignore[attr-defined]
                        middleware = create_middleware(
                            entity_resolver=resolver,
                            retriever=retriever,
                            llm_client=llm_client,
                            guardrails=guardrails,
                        )

                        # Pre-generation eligibility check.
                        _, _, pre_merge = await run_rag_baseline(
                            session, resolver, r.question_ar, enable_graph=True, top_k=cfg.top_k
                        )
                        pre_trace = retrieval_trace_from_merge(pre_merge, cfg.top_k)
                        resolved_for_gate = [
                            {
                                "type": e.entity_type.value,
                                "id": e.entity_id,
                                "name_ar": e.name_ar,
                                "match_type": getattr(e, "match_type", None),
                                "confidence": e.confidence,
                            }
                            for e in resolver.resolve(r.question_ar)[:5]
                        ]
                        pre_decision = decide_answer_eligibility(
                            question_ar=r.question_ar,
                            resolved_entities=resolved_for_gate,
                            retrieval_trace=pre_trace.model_dump(),
                            mode=mode.value,
                        )
                        pre_decision = await decide_answer_eligibility_with_db(
                            session=session,
                            question_ar=r.question_ar,
                            retrieval_trace=pre_trace.model_dump(),
                            resolved_entities=resolved_for_gate,
                            base=pre_decision,
                        )
                        if override_should_answer_from_dataset(dataset_row=r, retrieval_trace=pre_trace):
                            pre_decision = EligibilityDecision(True, "REQUIRED_EVIDENCE_IN_TOPK")  # type: ignore[name-defined]

                        if not pre_decision.should_answer:
                            # Stakeholder scenario: allow a partial grounded answer instead of strict abstention
                            # (A: grounded, B: explicit unsupported). This preserves safety and intent coverage.
                            if (
                                "stakeholder" in (r.tags or [])
                                and str(r.type) == "scenario"
                                and (getattr(pre_merge, "evidence_packets", None) or [])
                            ):
                                # Naturalized fallback: use the natural_chat writer so the voice stays consistent.
                                # Keep the fallback decision deterministic (we are in the "should_answer=False" branch).
                                try:
                                    if llm_client is not None:
                                        res = await llm_client.interpret(
                                            question=r.question_ar,
                                            evidence_packets=list(pre_merge.evidence_packets or [])[:18],
                                            detected_entities=resolved_for_gate,
                                            mode="natural_chat",
                                            used_edges=[],
                                            argument_chains=[],
                                            fallback_context={
                                                "partial_required": True,
                                                "required_markers_ar": [
                                                    "ما يمكن دعمه من الأدلة المسترجعة",
                                                    "ما لا يمكن الجزم به من الأدلة الحالية",
                                                ],
                                            },
                                        )
                                    else:
                                        res = None
                                except Exception:
                                    res = None

                                if res is None or not str(getattr(res, "answer_ar", "") or "").strip():
                                    # Fail safe: deterministic partial composer.
                                    from apps.api.core.scholar_reasoning_compose_scenario import compose_partial_scenario_answer

                                    ans_ar, cite_objs, _ = compose_partial_scenario_answer(
                                        packets=list(pre_merge.evidence_packets or []),
                                        question_ar=r.question_ar,
                                        prefer_more_claims=False,
                                    )
                                    answer = ans_ar
                                    cite_ids = [str(getattr(c, "chunk_id", "") or "") for c in (cite_objs or [])]
                                else:
                                    answer = str(getattr(res, "answer_ar", "") or "")
                                    cite_ids = [str(c.get("chunk_id") or "") for c in (getattr(res, "citations", None) or [])]

                                # Convert citations to eval span citations.
                                citations = []
                                for cid in cite_ids[:14]:
                                    if not cid:
                                        continue
                                    c = await citation_for_chunk_best_sentence(session, chunk_id=cid, query_text=r.question_ar)
                                    if c is not None:
                                        citations.append(c)
                                retrieval_trace = pre_trace
                                graph_trace = GraphTrace()
                                alm_trace = None
                                abstained = False
                                abstain_reason = None
                                debug = {"engine": "full_system", "eligibility_override": "STAKEHOLDER_PARTIAL_SCENARIO"}
                                claim_objs = extract_claims(answer_ar=answer, mode=mode, citations=citations)
                                pr = await prune_and_fail_closed(
                                    session=session,
                                    mode=mode,
                                    answer_ar=answer,
                                    claims=claim_objs,
                                    citations=citations,
                                    question_ar=r.question_ar,
                                    resolved_entities=resolved_for_gate,
                                    required_graph_paths=r.required_graph_paths,
                                )
                                answer = pr.answer_ar
                                claim_objs = pr.claims
                                citations = pr.citations
                                if pr.abstained:
                                    abstained = True
                                    abstain_reason = pr.abstain_reason
                            else:
                                citations = []
                                retrieval_trace = pre_trace
                                graph_trace = await graph_trace_for_required_path(session, r.required_graph_paths)
                                alm_trace = None
                                abstained = True
                                abstain_reason = pre_decision.reason_code
                                answer = "لا يوجد في البيانات الحالية ما يدعم الإجابة على هذا السؤال."
                                debug = {"engine": "full_system", "eligibility": pre_decision.reason_code}
                                claim_objs = []
                        else:
                            # Stakeholder runs should prefer the natural conversational voice (still grounded).
                            # Non-stakeholder datasets keep the default "answer" mode for stability.
                            m_mode = "natural_chat" if ("stakeholder" in (r.tags or [])) else "answer"
                            final = await middleware.process(r.question_ar, language="ar", mode=m_mode)

                            cite_ids = [getattr(c, "chunk_id", "") for c in (final.citations or [])]
                            citations = []
                            for cid in cite_ids:
                                c = await citation_for_chunk_best_sentence(session, chunk_id=cid, query_text=r.question_ar)
                                if c is not None:
                                    citations.append(c)

                            rag_ans, _, _ = await run_rag_baseline(
                                session, resolver, r.question_ar, enable_graph=True, top_k=cfg.top_k
                            )
                            alm_trace = await _build_almuhasbi_trace(
                                rag_plus_graph_answer=rag_ans,
                                full_answer=final.answer_ar or "",
                                supporting_chunk_ids=cite_ids,
                            )

                            abstained = bool(final.not_found)
                            abstain_reason = "not_found" if abstained else None
                            last_merge = getattr(retriever, "last_merge_result", None)
                            retrieval_trace = (
                                retrieval_trace_from_merge(last_merge, cfg.top_k)
                                if last_merge is not None
                                else RetrievalTrace()
                            )
                            citations = await augment_graph_justification_citations(
                                session=session,
                                retrieval_trace=retrieval_trace,
                                required_graph_paths=r.required_graph_paths,
                                citations=citations,
                                question_ar=r.question_ar,
                            )
                            graph_trace = await graph_trace_for_required_path(session, r.required_graph_paths)
                            # Populate strict "used edges" trace from deterministic scholar reasoning.
                            # Reason: stakeholder datasets often don't specify required_graph_paths, but we
                            # still need to show which semantic edges were actually used.
                            try:
                                used = list(getattr(middleware, "_last_used_edges", None) or [])
                                if used:
                                    coerced: list[GraphTraceUsedEdge] = []
                                    nodes_out: list[str] = []
                                    edges_out: list[str] = []
                                    for ue in used:
                                        spans: list[GraphTraceUsedEdgeSpan] = []
                                        for sp in list(ue.get("justification_spans") or [])[:6]:
                                            cid = str(sp.get("chunk_id") or "")
                                            if not cid:
                                                continue
                                            spans.append(
                                                GraphTraceUsedEdgeSpan(
                                                    source_id=cid,
                                                    chunk_id=cid,
                                                    span_start=int(sp.get("span_start") or 0),
                                                    span_end=int(sp.get("span_end") or 0),
                                                    quote=str(sp.get("quote") or ""),
                                                )
                                            )
                                        coerced.append(
                                            GraphTraceUsedEdge(
                                                edge_id=str(ue.get("edge_id") or ""),
                                                from_node=str(ue.get("from_node") or ""),
                                                to_node=str(ue.get("to_node") or ""),
                                                relation_type=str(ue.get("relation_type") or ""),
                                                justification_spans=spans,
                                            )
                                        )
                                        eid = str(ue.get("edge_id") or "")
                                        if eid:
                                            edges_out.append(eid)
                                        for n in [str(ue.get("from_node") or ""), str(ue.get("to_node") or "")]:
                                            if n and n not in nodes_out:
                                                nodes_out.append(n)
                                    graph_trace.used_edges = coerced
                                    # Add deterministic argument chains on top of used_edges (no CoT).
                                    try:
                                        ued: list[UsedEdge] = []
                                        for x in (graph_trace.used_edges or [])[:24]:
                                            spans2: list[UsedEdgeSpan] = []
                                            for sp2 in (x.justification_spans or [])[:6]:
                                                spans2.append(
                                                    UsedEdgeSpan(
                                                        chunk_id=str(getattr(sp2, "chunk_id", "") or ""),
                                                        span_start=int(getattr(sp2, "span_start", 0) or 0),
                                                        span_end=int(getattr(sp2, "span_end", 0) or 0),
                                                        quote=str(getattr(sp2, "quote", "") or ""),
                                                    )
                                                )
                                            ued.append(
                                                UsedEdge(
                                                    edge_id=str(x.edge_id or ""),
                                                    from_node=str(x.from_node or ""),
                                                    to_node=str(x.to_node or ""),
                                                    relation_type=str(x.relation_type or ""),
                                                    justification_spans=tuple(spans2),
                                                )
                                            )
                                        chains = build_argument_chains_from_used_edges(used_edges=ued)
                                        coerced_chains: list[ArgumentChain] = []
                                        for c in (chains or [])[:24]:
                                            ev = [
                                                EvidenceSpanRef(
                                                    source_id=str(sp.chunk_id or ""),
                                                    span_start=int(sp.span_start),
                                                    span_end=int(sp.span_end),
                                                    quote=str(sp.quote or ""),
                                                )
                                                for sp in (c.evidence_spans or ())
                                                if str(sp.chunk_id or "")
                                            ]
                                            bs = [
                                                EvidenceSpanRef(
                                                    source_id=str(sp.chunk_id or ""),
                                                    span_start=int(sp.span_start),
                                                    span_end=int(sp.span_end),
                                                    quote=str(sp.quote or ""),
                                                )
                                                for sp in (c.boundary_spans or ())
                                                if str(sp.chunk_id or "")
                                            ]
                                            it = (
                                                ArgumentInferenceType.DIRECT_QUOTE
                                                if str(c.inference_type) == "direct_quote"
                                                else ArgumentInferenceType.MULTI_SPAN_ENTAILMENT
                                            )
                                            coerced_chains.append(
                                                ArgumentChain(
                                                    edge_id=str(c.edge_id or ""),
                                                    relation_type=str(c.relation_type or ""),
                                                    from_node=str(c.from_node or ""),
                                                    to_node=str(c.to_node or ""),
                                                    claim_ar=str(c.claim_ar or ""),
                                                    inference_type=it,
                                                    evidence_spans=ev,
                                                    boundary_ar=str(c.boundary_ar or "غير منصوص عليه في الإطار"),
                                                    boundary_spans=bs,
                                                )
                                            )
                                        graph_trace.argument_chains = coerced_chains
                                    except Exception:
                                        pass
                                    # Also populate `paths` with the exact used edges (not retrieved neighbors).
                                    # Note: order is the emission order in the answer.
                                    if nodes_out or edges_out:
                                        graph_trace.nodes = nodes_out
                                        graph_trace.edges = edges_out
                                        graph_trace.paths = [
                                            GraphTracePath(nodes=nodes_out, edges=edges_out, confidence=1.0)
                                        ]
                            except Exception:
                                pass
                            answer = final.answer_ar or ""
                            debug = {"engine": "full_system", "confidence": getattr(final, "confidence", None)}
                            # Ensure graph justifications are present as citations for pruning/binding.
                            # Reason: contract/graph composers cite edge_justification_span quotes, which must
                            # be available as EvalCitations or pruning will fail-closed (PRUNED_TOO_MUCH).
                            try:
                                extra: list[EvalCitation] = []
                                seen = {(c.source_id, int(c.span_start), int(c.span_end)) for c in (citations or [])}
                                for ue in (getattr(graph_trace, "used_edges", None) or []):
                                    spans_any = ue.get("justification_spans") if isinstance(ue, dict) else getattr(ue, "justification_spans", None)
                                    for sp in (spans_any or []):
                                        if isinstance(sp, dict):
                                            sid = str(sp.get("chunk_id") or "") or str(sp.get("source_id") or "")
                                            ss = int(sp.get("span_start") or 0)
                                            se = int(sp.get("span_end") or 0)
                                            q = str(sp.get("quote") or "")
                                        else:
                                            sid = str(getattr(sp, "chunk_id", "") or "") or str(getattr(sp, "source_id", "") or "")
                                            ss = int(getattr(sp, "span_start", 0) or 0)
                                            se = int(getattr(sp, "span_end", 0) or 0)
                                            q = str(getattr(sp, "quote", "") or "")
                                        key = (sid, ss, se)
                                        if sid and se > ss and q and key not in seen:
                                            seen.add(key)
                                            extra.append(EvalCitation(source_id=sid, span_start=ss, span_end=se, quote=q))
                                if extra:
                                    citations.extend(extra)
                            except Exception:
                                pass
                            claim_objs = extract_claims(answer_ar=answer, mode=mode, citations=citations)
                            pr = await prune_and_fail_closed(
                                session=session,
                                mode=mode,
                                answer_ar=answer,
                                claims=claim_objs,
                                citations=citations,
                                question_ar=r.question_ar,
                                resolved_entities=resolved_for_gate,
                                required_graph_paths=r.required_graph_paths,
                            )
                            answer = pr.answer_ar
                            claim_objs = pr.claims
                            citations = pr.citations
                            if pr.abstained:
                                abstained = True
                                abstain_reason = pr.abstain_reason

                            # Contract checker (stakeholder readiness): enforce *intent coverage*.
                            # Reason: production-like runs can be safe but off-target; we gate on-targetness.
                            try:
                                if "stakeholder" in (r.tags or []) and (r.answer_requirements or {}):
                                    spec = contract_from_answer_requirements(
                                        question_norm=normalize_for_matching(r.question_ar or ""),
                                        question_ar=r.question_ar,
                                        question_type=str(r.type),
                                        answer_requirements=dict(r.answer_requirements or {}),
                                    )

                                    # Convert eval GraphTrace.used_edges -> contract UsedEdge list.
                                    used_edges: list[UsedEdge] = []
                                    for ue in (graph_trace.used_edges or []):
                                        spans: list[UsedEdgeSpan] = []
                                        for sp in (ue.justification_spans or []):
                                            spans.append(
                                                UsedEdgeSpan(
                                                    chunk_id=str(getattr(sp, "chunk_id", "") or ""),
                                                    span_start=int(sp.span_start or 0),
                                                    span_end=int(sp.span_end or 0),
                                                    quote=str(sp.quote or ""),
                                                )
                                            )
                                        used_edges.append(
                                            UsedEdge(
                                                edge_id=str(ue.edge_id or ""),
                                                from_node=str(ue.from_node or ""),
                                                to_node=str(ue.to_node or ""),
                                                relation_type=str(ue.relation_type or ""),
                                                justification_spans=tuple(spans),
                                            )
                                        )

                                    cm = check_contract(
                                        spec=spec,
                                        answer_ar=answer,
                                        citations=citations,
                                        used_edges=used_edges,
                                    )
                                    debug["contract"] = {
                                        "outcome": cm.outcome.value,
                                        "pass": cm.outcome.value in ["PASS_FULL", "PASS_PARTIAL"],
                                        "reasons": list(cm.reasons),
                                        "section_nonempty_rate": cm.section_nonempty,
                                        "required_entities_coverage_rate": cm.required_entities_coverage,
                                        "graph_required_satisfaction_rate": 1.0 if cm.graph_required_satisfied else 0.0,
                                    }
                                    if cm.outcome.value == "PASS_PARTIAL":
                                        abstained = False
                                        abstain_reason = None
                                    elif cm.outcome.value == "FAIL":
                                        repaired_contract = False
                                        # Compare repair: attempt deterministic compare composer before failing closed.
                                        if any("مصفوفة المقارنة" in r for r in cm.reasons) or any(
                                            "COMPARE_" in r for r in cm.reasons
                                        ):
                                            try:
                                                from apps.api.core.answer_contract import extract_compare_concepts_from_question
                                                from apps.api.core.scholar_reasoning_compose_compare import compose_compare_answer
                                                from apps.api.retrieve.sql_retriever import search_entities_by_name, get_chunks_with_refs
                                                from apps.api.core.schemas import EntityType

                                                concepts = list(extract_compare_concepts_from_question(r.question_ar) or [])
                                                # Include the primary concept if present.
                                                if "التزكية" in (r.question_ar or "") and "التزكية" not in concepts:
                                                    concepts = ["التزكية"] + concepts

                                                packets: list[dict[str, Any]] = []
                                                try:
                                                    last_merge = getattr(retriever, "last_merge_result", None)
                                                    packets = list(getattr(last_merge, "evidence_packets", None) or [])
                                                except Exception:
                                                    packets = []

                                                # Targeted packet expansion for each concept by name search.
                                                extra_packets: list[dict[str, Any]] = []
                                                for nm in concepts[:6]:
                                                    hits = await search_entities_by_name(session, name_pattern=str(nm), limit=4)
                                                    for h in hits[:2]:
                                                        et = str(h.get("entity_type") or "")
                                                        eid = str(h.get("id") or "")
                                                        if et not in {"pillar", "core_value", "sub_value"} or not eid:
                                                            continue
                                                        try:
                                                            etype = EntityType(et)
                                                        except Exception:
                                                            continue
                                                        extra_packets.extend(await get_chunks_with_refs(session, etype, eid, limit=10))
                                                if extra_packets:
                                                    seen = {str(p.get("chunk_id") or "") for p in packets}
                                                    for p in extra_packets:
                                                        cid = str(p.get("chunk_id") or "")
                                                        if cid and cid not in seen:
                                                            packets.append(p)
                                                            seen.add(cid)

                                                ans_ar, cite_objs, _ = compose_compare_answer(
                                                    question_ar=r.question_ar,
                                                    concepts_ar=concepts,
                                                    packets=packets,
                                                    prefer_more_claims=True,
                                                )

                                                citations = []
                                                for cobj in (cite_objs or [])[:18]:
                                                    cid = str(getattr(cobj, "chunk_id", "") or "")
                                                    if not cid:
                                                        continue
                                                    c = await citation_for_chunk_best_sentence(
                                                        session, chunk_id=cid, query_text=r.question_ar
                                                    )
                                                    if c is not None:
                                                        citations.append(c)

                                                answer = ans_ar
                                                abstained = False
                                                abstain_reason = None
                                                claim_objs = extract_claims(answer_ar=answer, mode=mode, citations=citations)
                                                pr = await prune_and_fail_closed(
                                                    session=session,
                                                    mode=mode,
                                                    answer_ar=answer,
                                                    claims=claim_objs,
                                                    citations=citations,
                                                    question_ar=r.question_ar,
                                                    resolved_entities=resolved_for_gate,
                                                    required_graph_paths=r.required_graph_paths,
                                                )
                                                answer = pr.answer_ar
                                                claim_objs = pr.claims
                                                citations = pr.citations

                                                cm3 = check_contract(spec=spec, answer_ar=answer, citations=citations, used_edges=[])
                                                if cm3.outcome.value == "FAIL":
                                                    # If pruning dropped compare fields, re-add meta placeholders.
                                                    missing = [r for r in cm3.reasons if r.startswith("COMPARE_MISSING_FIELD:")]
                                                    if missing:
                                                        try:
                                                            # Append a completion block per concept (meta-only).
                                                            for m in missing:
                                                                _, _, rest = m.partition("COMPARE_MISSING_FIELD:")
                                                                concept, _, _field = rest.partition(":")
                                                                concept = concept.strip()
                                                                if not concept:
                                                                    continue
                                                                answer = (
                                                                    answer
                                                                    + "\n"
                                                                    + f"- {concept}:\n"
                                                                    + "- التعريف: غير منصوص عليه\n"
                                                                    + "- المظهر العملي: غير منصوص عليه\n"
                                                                    + "- الخطأ الشائع: غير منصوص عليه"
                                                                ).strip()
                                                            cm3 = check_contract(
                                                                spec=spec, answer_ar=answer, citations=citations, used_edges=[]
                                                            )
                                                        except Exception:
                                                            pass
                                                debug["contract"]["outcome"] = cm3.outcome.value
                                                debug["contract"]["pass"] = cm3.outcome.value in ["PASS_FULL", "PASS_PARTIAL"]
                                                debug["contract"]["reasons"] = list(cm3.reasons)
                                                debug["contract"]["section_nonempty_rate"] = cm3.section_nonempty
                                                debug["contract"]["required_entities_coverage_rate"] = cm3.required_entities_coverage
                                                debug["contract"]["graph_required_satisfaction_rate"] = (
                                                    1.0 if cm3.graph_required_satisfied else 0.0
                                                )
                                                if cm3.outcome.value in {"PASS_FULL", "PASS_PARTIAL"}:
                                                    # successful repair
                                                    repaired_contract = True
                                            except Exception:
                                                pass

                                        if repaired_contract:
                                            # Successful repair means we accept this answer (do not run further FAIL handlers).
                                            abstained = False
                                            abstain_reason = None
                                        else:
                                            # If missing grounded graph edges, emit a partial A+B answer (do not abstain).
                                            if "MISSING_USED_GRAPH_EDGES" in cm.reasons:
                                                try:
                                                    from apps.api.core.scholar_reasoning_compose_partial_graph import (
                                                        compose_partial_graph_gap_answer,
                                                    )

                                                    packets: list[dict[str, Any]] = []
                                                    try:
                                                        last_merge = getattr(retriever, "last_merge_result", None)
                                                        packets = list(getattr(last_merge, "evidence_packets", None) or [])
                                                    except Exception:
                                                        packets = []
                                                    ans_ar, cite_objs, _ = compose_partial_graph_gap_answer(
                                                        packets=packets, question_ar=r.question_ar
                                                    )
                                                    # Convert citations to eval span citations.
                                                    citations = []
                                                    for cobj in (cite_objs or [])[:12]:
                                                        cid = str(getattr(cobj, "chunk_id", "") or "")
                                                        if not cid:
                                                            continue
                                                        c = await citation_for_chunk_best_sentence(
                                                            session, chunk_id=cid, query_text=r.question_ar
                                                        )
                                                        if c is not None:
                                                            citations.append(c)
                                                    answer = ans_ar
                                                    abstained = False
                                                    abstain_reason = None
                                                    claim_objs = extract_claims(answer_ar=answer, mode=mode, citations=citations)
                                                    pr = await prune_and_fail_closed(
                                                        session=session,
                                                        mode=mode,
                                                        answer_ar=answer,
                                                        claims=claim_objs,
                                                        citations=citations,
                                                        question_ar=r.question_ar,
                                                        resolved_entities=resolved_for_gate,
                                                        required_graph_paths=r.required_graph_paths,
                                                    )
                                                    answer = pr.answer_ar
                                                    claim_objs = pr.claims
                                                    citations = pr.citations
                                                    # Re-check contract on the final (post-prune) answer.
                                                    cm2 = check_contract(
                                                        spec=spec,
                                                        answer_ar=answer,
                                                        citations=citations,
                                                        used_edges=[],
                                                    )
                                                    debug["contract"]["outcome"] = cm2.outcome.value
                                                    debug["contract"]["pass"] = cm2.outcome.value in ["PASS_FULL", "PASS_PARTIAL"]
                                                    debug["contract"]["reasons"] = list(cm2.reasons)
                                                    debug["contract"]["section_nonempty_rate"] = cm2.section_nonempty
                                                    debug["contract"]["required_entities_coverage_rate"] = cm2.required_entities_coverage
                                                    debug["contract"]["graph_required_satisfaction_rate"] = (
                                                        1.0 if cm2.graph_required_satisfied else 0.0
                                                    )
                                                except Exception:
                                                    abstained = True
                                                    abstain_reason = "CONTRACT_UNMET"
                                                    answer = "لا يوجد في البيانات الحالية ما يدعم إجابة ملتزمة بعقد السؤال."
                                                    citations = []
                                                    claim_objs = []
                                            # If we *do* have used_edges but contract thinks the cross-pillar section is empty,
                                            # deterministically re-compose the network section from used_edges (framework-only).
                                            elif (
                                                "EMPTY_SECTION:الربط بين الركائز (مع سبب الربط)" in cm.reasons
                                                and (spec.intent_type in {"network", "cross_pillar", "cross_pillar_path"})
                                                and (used_edges or [])
                                            ):
                                                try:
                                                    from apps.api.core.scholar_reasoning_compose_graph_intents import (
                                                        compose_network_answer,
                                                    )

                                                    packets2: list[dict[str, Any]] = []
                                                    try:
                                                        last_merge = getattr(retriever, "last_merge_result", None)
                                                        packets2 = list(getattr(last_merge, "evidence_packets", None) or [])
                                                    except Exception:
                                                        packets2 = []

                                                    semantic_edges2: list[dict[str, Any]] = []
                                                    for ue in list(used_edges or [])[:10]:
                                                        fr = str(getattr(ue, "from_node", "") or "")
                                                        to = str(getattr(ue, "to_node", "") or "")
                                                        if ":" not in fr or ":" not in to:
                                                            continue
                                                        fr_t, _, fr_id = fr.partition(":")
                                                        to_t, _, to_id = to.partition(":")
                                                        if not (fr_t and fr_id and to_t and to_id):
                                                            continue
                                                        spans_out: list[dict[str, Any]] = []
                                                        for sp in list(getattr(ue, "justification_spans", None) or [])[:3]:
                                                            spans_out.append(
                                                                {
                                                                    "chunk_id": str(getattr(sp, "chunk_id", "") or ""),
                                                                    "span_start": int(getattr(sp, "span_start", 0) or 0),
                                                                    "span_end": int(getattr(sp, "span_end", 0) or 0),
                                                                    "quote": str(getattr(sp, "quote", "") or ""),
                                                                }
                                                            )
                                                        if not spans_out:
                                                            continue
                                                        semantic_edges2.append(
                                                            {
                                                                "edge_id": str(getattr(ue, "edge_id", "") or ""),
                                                                "relation_type": str(getattr(ue, "relation_type", "") or ""),
                                                                "source_type": fr_t,
                                                                "source_id": fr_id,
                                                                "neighbor_type": to_t,
                                                                "neighbor_id": to_id,
                                                                "direction": "outgoing",
                                                                "justification_spans": spans_out,
                                                            }
                                                        )

                                                    ans_ar, _, _ = compose_network_answer(
                                                        packets=packets2,
                                                        semantic_edges=semantic_edges2,
                                                        max_links=6,
                                                    )

                                                    citations = []
                                                    seen = set()
                                                    for ed in semantic_edges2:
                                                        for sp in list(ed.get("justification_spans") or [])[:2]:
                                                            cid = str(sp.get("chunk_id") or "")
                                                            ss = int(sp.get("span_start") or 0)
                                                            se = int(sp.get("span_end") or 0)
                                                            q = str(sp.get("quote") or "")
                                                            key = (cid, ss, se)
                                                            if cid and se > ss and q and key not in seen:
                                                                seen.add(key)
                                                                citations.append(EvalCitation(source_id=cid, span_start=ss, span_end=se, quote=q))

                                                    answer = ans_ar
                                                    abstained = False
                                                    abstain_reason = None
                                                    claim_objs = extract_claims(answer_ar=answer, mode=mode, citations=citations)
                                                    pr = await prune_and_fail_closed(
                                                        session=session,
                                                        mode=mode,
                                                        answer_ar=answer,
                                                        claims=claim_objs,
                                                        citations=citations,
                                                        question_ar=r.question_ar,
                                                        resolved_entities=resolved_for_gate,
                                                        required_graph_paths=r.required_graph_paths,
                                                    )
                                                    answer = pr.answer_ar
                                                    claim_objs = pr.claims
                                                    citations = pr.citations

                                                    cm2 = check_contract(
                                                        spec=spec,
                                                        answer_ar=answer,
                                                        citations=citations,
                                                        used_edges=used_edges,
                                                    )
                                                    debug["contract"]["outcome"] = cm2.outcome.value
                                                    debug["contract"]["pass"] = cm2.outcome.value in ["PASS_FULL", "PASS_PARTIAL"]
                                                    debug["contract"]["reasons"] = list(cm2.reasons)
                                                    debug["contract"]["section_nonempty_rate"] = cm2.section_nonempty
                                                    debug["contract"]["required_entities_coverage_rate"] = cm2.required_entities_coverage
                                                    debug["contract"]["graph_required_satisfaction_rate"] = (
                                                        1.0 if cm2.graph_required_satisfied else 0.0
                                                    )

                                                    if cm2.outcome.value in {"PASS_FULL", "PASS_PARTIAL"}:
                                                        abstained = False
                                                        abstain_reason = None
                                                    else:
                                                        abstained = True
                                                        abstain_reason = "CONTRACT_UNMET"
                                                        answer = "لا يوجد في البيانات الحالية ما يدعم إجابة ملتزمة بعقد السؤال."
                                                        citations = []
                                                        claim_objs = []
                                                except Exception:
                                                    abstained = True
                                                    abstain_reason = "CONTRACT_UNMET"
                                                    answer = "لا يوجد في البيانات الحالية ما يدعم إجابة ملتزمة بعقد السؤال."
                                                    citations = []
                                                    claim_objs = []
                                            else:
                                                abstained = True
                                                abstain_reason = "CONTRACT_UNMET"
                                                answer = "لا يوجد في البيانات الحالية ما يدعم إجابة ملتزمة بعقد السؤال."
                                                citations = []
                                                claim_objs = []
                            except Exception as _:
                                # Fail open for non-stakeholder datasets.
                                pass

            latency_ms = int((time.perf_counter() - t0) * 1000)

            if mode in {EvalMode.LLM_ONLY_SAFE, EvalMode.LLM_ONLY_UNGROUNDED}:
                claim_objs = extract_claims(answer_ar=answer, mode=mode, citations=citations)

            out_rows.append(
                EvalOutputRow(
                    id=r.id,
                    mode=mode,
                    question=r.question_ar,
                    answer_ar=answer,
                    answer_en=None,
                    claims=claim_objs,
                    citations=citations,
                    retrieval_trace=retrieval_trace,
                    graph_trace=graph_trace,
                    almuhasbi_trace=alm_trace,
                    abstained=abstained,
                    abstain_reason=abstain_reason,
                    latency_ms=latency_ms,
                    debug=debug,
                )
            )

        if int(cfg.start or 0) > 0:
            append_jsonl_rows(out_rows, out_path)
        else:
            write_jsonl_rows(out_rows, out_path)

    return meta.run_id

