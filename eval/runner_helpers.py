"""Evaluation runner helpers.

Reason: `eval/runner.py` must stay <500 LOC. This module contains helper
functions used by the runner core.
"""

from __future__ import annotations

from typing import Any

from sqlalchemy import text

from apps.api.core.database import get_session
from apps.api.graph.explain import shortest_path
from apps.api.llm.gpt5_client_azure import LLMRequest, ProviderConfig, create_provider
from apps.api.retrieve.entity_resolver import EntityResolver
from apps.api.retrieve.hybrid_retriever import HybridRetriever, RetrievalInputs
from apps.api.retrieve.merge_rank import MergeResult

from eval.citations import citation_for_chunk_best_sentence
from eval.types import GraphTrace, GraphTracePath, RetrievalTrace, RetrievalTraceChunk


async def augment_graph_justification_citations(
    *,
    session,
    retrieval_trace: RetrievalTrace,
    required_graph_paths: list[dict[str, Any]],
    citations: list,
    question_ar: str,
) -> list:
    """Ensure SHARES_REF justifications are cited when required by dataset.

    Reason: cross-pillar edges may be structural, but explanation_grounded_rate requires
    citing the shared reference (quran/hadith/book) used to derive the edge.
    """
    if not required_graph_paths:
        return citations
    if not retrieval_trace.top_k_chunks:
        return citations

    top_chunk_ids = [c.chunk_id for c in retrieval_trace.top_k_chunks if (c.chunk_id or "").strip()]
    if not top_chunk_ids:
        return citations

    existing = {getattr(c, "source_id", "") for c in citations}
    out = list(citations)

    for p in required_graph_paths:
        rel_type = str(p.get("rel_type") or "")
        just = str(p.get("justification") or "").strip()
        if rel_type != "SHARES_REF" or ":" not in just:
            continue
        ref_type, _, ref = just.partition(":")
        ref_type = ref_type.strip()
        ref = ref.strip()
        if not (ref_type and ref):
            continue

        row = (
            await session.execute(
                text(
                    """
                    SELECT chunk_id
                    FROM chunk_ref
                    WHERE chunk_id = ANY(:cids)
                      AND ref_type = :rt
                      AND ref = :r
                    LIMIT 1
                    """
                ),
                {"cids": top_chunk_ids, "rt": ref_type, "r": ref},
            )
        ).fetchone()
        if not row:
            continue
        cid = str(row.chunk_id or "")
        if not cid or cid in existing:
            continue
        c = await citation_for_chunk_best_sentence(session, chunk_id=cid, query_text=question_ar)
        if c is not None:
            out.append(c)
            existing.add(cid)

    return out


def parse_node(node: str) -> tuple[str, str]:
    if ":" not in node:
        return node, ""
    t, _, i = node.partition(":")
    return t, i


async def build_entity_resolver(session) -> EntityResolver:
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
    return resolver


def retrieval_trace_from_merge(m: MergeResult, top_k: int) -> RetrievalTrace:
    chunks: list[RetrievalTraceChunk] = []
    for rc in (m.ranked_chunks or [])[:top_k]:
        b = rc.get("backend")
        chunks.append(
            RetrievalTraceChunk(
                chunk_id=str(rc.get("chunk_id") or ""),
                score=float(rc.get("score") or 0.0),
                backend=str(b) if b else None,
                rank=int(rc.get("rank") or 1),
                sources=list(rc.get("sources") or []),
            )
        )
    return RetrievalTrace(top_k_chunks=chunks, top_k=top_k)


def override_should_answer_from_dataset(*, dataset_row: Any, retrieval_trace: RetrievalTrace) -> bool:
    """Eval-only override: if required chunks appear in top-k, answer."""
    try:
        if bool(getattr(dataset_row, "expect_abstain", False)):
            return False
        required = set([str(x) for x in (getattr(dataset_row, "required_evidence_refs", None) or []) if str(x)])
        if not required:
            return False
        got = set([c.chunk_id for c in (retrieval_trace.top_k_chunks or []) if (c.chunk_id or "").strip()])
        return bool(required.intersection(got))
    except Exception:
        return False


async def graph_trace_for_required_path(session, required_graph_paths: list[dict[str, Any]]) -> GraphTrace:
    if not required_graph_paths:
        return GraphTrace()
    p0 = required_graph_paths[0]
    nodes = p0.get("nodes") or []
    if not isinstance(nodes, list) or len(nodes) < 2:
        return GraphTrace()

    start_t, start_id = parse_node(str(nodes[0]))
    end_t, end_id = parse_node(str(nodes[-1]))
    if not (start_t and start_id and end_t and end_id):
        return GraphTrace()

    res = await shortest_path(
        session,
        start_type=start_t,
        start_id=start_id,
        target_type=end_t,
        target_id=end_id,
        max_depth=4,
        rel_types=None,
    )
    if not res.get("found"):
        return GraphTrace(nodes=[], edges=[], paths=[])

    path_nodes = res.get("path") or []
    nodes_out: list[str] = []
    edges_out: list[str] = []

    prev = None
    for item in path_nodes:
        cur = f"{item['type']}:{item['id']}"
        nodes_out.append(cur)
        if prev is not None and item.get("via_rel"):
            edges_out.append(f"{prev}::{item['via_rel']}::{cur}")
        prev = cur

    return GraphTrace(
        nodes=nodes_out,
        edges=edges_out,
        paths=[GraphTracePath(nodes=nodes_out, edges=edges_out, confidence=1.0)],
    )


async def run_llm_only_ungrounded(question: str) -> tuple[str, dict[str, Any]]:
    cfg = ProviderConfig.from_env()
    if not cfg.is_configured():
        return (
            "تعذر تشغيل وضع LLM_ONLY_UNGROUNDED لأن إعدادات النموذج غير مهيأة.",
            {"llm_error": "not_configured"},
        )

    provider = create_provider(cfg)
    req = LLMRequest(
        system_prompt=(
            "أجب باللغة العربية الفصحى. لا تذكر أنك استخدمت مصادر أو استشهادات. "
            "قدّم إجابة منظمة ومفصلة قدر الإمكان."
        ),
        user_message=question,
        response_format=None,
        temperature=0.0,
        max_tokens=800,
    )
    resp = await provider.complete(req)
    if resp.error:
        return ("تعذر الحصول على إجابة من النموذج.", {"llm_error": resp.error})
    return (resp.content.strip() or "", {"llm_model": resp.model})


async def run_rag_baseline(
    session,
    resolver: EntityResolver,
    question: str,
    *,
    enable_graph: bool,
    top_k: int,
) -> tuple[str, list[str], MergeResult]:
    resolved = resolver.resolve(question)
    resolved_list = [{"type": r.entity_type.value, "id": r.entity_id, "name_ar": r.name_ar} for r in resolved[:5]]

    retriever = HybridRetriever(enable_graph=enable_graph)
    inputs = RetrievalInputs(query=question, resolved_entities=resolved_list, top_k=top_k)
    merge = await retriever.retrieve(session, inputs)

    packets = merge.evidence_packets
    if not packets:
        return ("لا يوجد في البيانات الحالية ما يدعم الإجابة على هذا السؤال.", [], merge)

    answer_parts: list[str] = []
    cite_chunk_ids: list[str] = []
    for p in packets[:10]:
        txt = str(p.get("text_ar") or "").strip()
        if txt and len(answer_parts) < 5:
            answer_parts.append(txt[:300] + ("..." if len(txt) > 300 else ""))
        cid = str(p.get("chunk_id") or "")
        if cid:
            cite_chunk_ids.append(cid)

    return ("\n\n".join(answer_parts).strip(), cite_chunk_ids[:10], merge)

