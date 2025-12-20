"""
Hybrid Retrieval Pipeline

Orchestrates:
- Entity-first SQL retrieval
- Graph expansion (edges table)
- Vector retrieval (pgvector) when embeddings are available

Returns Evidence Packets (contract) via MergeRanker.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Optional

from sqlalchemy.ext.asyncio import AsyncSession

from apps.api.core.schemas import EntityType
from apps.api.retrieve.merge_rank import MergeRanker, MergeResult
from apps.api.retrieve.sql_retriever import get_chunks_with_refs
from apps.api.retrieve.graph_retriever import expand_graph
from apps.api.retrieve.vector_retriever import VectorRetriever
from apps.api.retrieve.reranker import Reranker, create_reranker_from_env


@dataclass
class RetrievalInputs:
    """Inputs to hybrid retrieval."""

    query: str
    resolved_entities: list[dict[str, Any]]
    top_k: int = 10
    graph_depth: int = 2


class HybridRetriever:
    """Hybrid retriever implementation."""

    def __init__(
        self,
        merge_ranker: Optional[MergeRanker] = None,
        vector_retriever: Optional[VectorRetriever] = None,
        reranker: Optional[Reranker] = None,
        enable_vector: bool = True,
        enable_graph: bool = True,
    ):
        self.merge_ranker = merge_ranker or MergeRanker(max_packets=25)
        self.vector_retriever = vector_retriever or VectorRetriever()
        self.reranker: Reranker = reranker or create_reranker_from_env()
        self.enable_vector = enable_vector
        self.enable_graph = enable_graph

    def _collect_ref_node_ids(self, packets: list[dict[str, Any]], max_refs: int = 12) -> list[str]:
        """
        Collect ref-node IDs from evidence packets.

        Ref node identity is '<type>:<ref>' (e.g., 'quran:البقرة:255').

        Reason: enables ref-driven graph expansion even when entity names aren't
        explicitly mentioned in the user query.
        """
        ref_ids: list[str] = []
        seen: set[str] = set()

        for p in packets:
            refs = p.get("refs") or []
            if not isinstance(refs, list):
                continue
            for r in refs:
                if not isinstance(r, dict):
                    continue
                t = r.get("type")
                ref = r.get("ref")
                if not t or not ref:
                    continue
                rid = f"{t}:{ref}"
                if rid not in seen:
                    seen.add(rid)
                    ref_ids.append(rid)
                if len(ref_ids) >= max_refs:
                    return ref_ids
        return ref_ids

    async def _expand_via_refs(
        self,
        session: AsyncSession,
        ref_node_ids: list[str],
        depth: int = 2,
        per_entity_limit: int = 6,
    ) -> list[dict[str, Any]]:
        """
        Expand graph starting from 'ref' nodes to reach related entities, then fetch their chunks.

        The traversal relies on:
        - entity ─MENTIONS_REF→ ref
        - evidence ─REFERS_TO→ ref
        """
        graph_packets: list[dict[str, Any]] = []
        for rid in ref_node_ids:
            neighbors = await expand_graph(
                session,
                entity_type="ref",
                entity_id=rid,
                depth=depth,
                    relationship_types=[
                        "MENTIONS_REF",
                        "REFERS_TO",
                        "SHARES_REF",
                        "SAME_NAME",
                        "CONTAINS",
                        "SUPPORTED_BY",
                        "SCHOLAR_LINK",
                    ],
            )
            for n in neighbors:
                n_type = n.get("neighbor_type")
                n_id = n.get("neighbor_id")
                if not n_type or not n_id:
                    continue
                # Only fetch chunks for well-defined entities
                try:
                    n_et = EntityType(n_type)
                except Exception:
                    continue
                packets = await get_chunks_with_refs(session, n_et, n_id, limit=per_entity_limit)
                for p in packets:
                    p["depth"] = n.get("depth", 1)
                    p["via_ref"] = rid
                graph_packets.extend(packets)
        return graph_packets

    async def retrieve(
        self,
        session: AsyncSession,
        inputs: RetrievalInputs,
        reranker_override: Optional[bool] = None,
    ) -> MergeResult:
        """
        Retrieve evidence packets using all enabled sources.

        Notes:
        - Vector retrieval is best-effort; if embeddings are not configured it may return none.
        - Graph expansion only uses approved edges (graph_retriever default).
        """
        sql_results: list[dict[str, Any]] = []
        graph_results: list[dict[str, Any]] = []
        vector_results: list[dict[str, Any]] = []

        inferred_entities: list[dict[str, Any]] = []

        # 1) Entity-first SQL retrieval
        for ent in inputs.resolved_entities:
            try:
                et = EntityType(ent["type"])
            except Exception:
                continue

            entity_id = ent.get("id") or ent.get("entity_id")
            if not entity_id:
                continue

            sql_results.extend(
                await get_chunks_with_refs(session, et, entity_id, limit=20)
            )

            # 2) Graph expansion (neighbors -> their chunks)
            if self.enable_graph:
                neighbors = await expand_graph(
                    session,
                    et,
                    entity_id,
                    depth=inputs.graph_depth,
                    relationship_types=[
                        "CONTAINS",
                        "SUPPORTED_BY",
                        "SHARES_REF",
                        "MENTIONS_REF",
                        "REFERS_TO",
                        "SAME_NAME",
                        "SCHOLAR_LINK",
                    ],
                )
                for n in neighbors:
                    n_type = n.get("neighbor_type")
                    n_id = n.get("neighbor_id")
                    if not n_type or not n_id:
                        continue
                    try:
                        n_et = EntityType(n_type)
                    except Exception:
                        continue
                    packets = await get_chunks_with_refs(session, n_et, n_id, limit=10)
                    # Tag depth so MergeRanker can down-weight if desired
                    for p in packets:
                        p["depth"] = n.get("depth", 1)
                    graph_results.extend(packets)

        # 3) Vector retrieval (best-effort)
        if self.enable_vector:
            try:
                vector_results = await self.vector_retriever.search(
                    session,
                    inputs.query,
                    top_k=inputs.top_k,
                )
            except Exception:
                vector_results = []

        # 3b) Infer entities from vector hits when explicit entity resolution fails.
        # Reason: Arabic users often ask via concepts not matching canonical names.
        # We can still "anchor" to the framework by using the entity_id/type in
        # the highest scoring chunks, then expand via graph.
        if not inputs.resolved_entities and vector_results:
            seen: set[tuple[str, str]] = set()
            for p in vector_results[: min(8, len(vector_results))]:
                et = p.get("entity_type")
                eid = p.get("entity_id")
                if not et or not eid:
                    continue
                key = (str(et), str(eid))
                if key in seen:
                    continue
                seen.add(key)
                inferred_entities.append(
                    {
                        "type": str(et),
                        "id": str(eid),
                        "name_ar": None,
                        "confidence": 0.45,
                        "match_type": "vector_inferred",
                    }
                )

        # 4) Ref-driven graph expansion (enterprise cross-pillar discovery)
        # Collect refs from what we already retrieved (SQL + vector) and traverse from 'ref' nodes.
        if self.enable_graph:
            ref_node_ids = self._collect_ref_node_ids(sql_results + vector_results)
            if ref_node_ids:
                graph_results.extend(await self._expand_via_refs(session, ref_node_ids, depth=2))

            # If we inferred entities from vector, expand from them too.
            for ent in inferred_entities:
                try:
                    et = EntityType(ent["type"])
                except Exception:
                    continue
                neighbors = await expand_graph(
                    session,
                    et,
                    ent["id"],
                    depth=inputs.graph_depth,
                    relationship_types=[
                        "CONTAINS",
                        "SUPPORTED_BY",
                        "SHARES_REF",
                        "MENTIONS_REF",
                        "REFERS_TO",
                        "SAME_NAME",
                        "SCHOLAR_LINK",
                    ],
                )
                for n in neighbors:
                    n_type = n.get("neighbor_type")
                    n_id = n.get("neighbor_id")
                    if not n_type or not n_id:
                        continue
                    try:
                        n_et = EntityType(n_type)
                    except Exception:
                        continue
                    packets = await get_chunks_with_refs(session, n_et, n_id, limit=6)
                    for p in packets:
                        p["depth"] = n.get("depth", 1)
                        p["via_entity"] = ent["id"]
                    graph_results.extend(packets)

        merged = self.merge_ranker.merge(
            sql_results=sql_results,
            vector_results=vector_results,
            graph_results=graph_results,
            resolved_entities=inputs.resolved_entities,
        )

        # Optional reranker pass (top-N reordering).
        # reranker_override: None=use default, True=force on, False=force off
        try:
            use_reranker = self.reranker.is_enabled() if reranker_override is None else reranker_override
            if self.reranker and use_reranker and merged.evidence_packets:
                alpha = float(os.getenv("RERANKER_ALPHA", "0.25") or 0.25)
                alpha = max(0.0, min(alpha, 1.0))
                # Build lookup from chunk_id -> base score/backends.
                base_by_id = {str(rc.get("chunk_id") or ""): float(rc.get("score") or 0.0) for rc in (merged.ranked_chunks or [])}
                sources_by_id = {str(rc.get("chunk_id") or ""): list(rc.get("sources") or []) for rc in (merged.ranked_chunks or [])}
                backend_by_id = {str(rc.get("chunk_id") or ""): str(rc.get("backend") or "") for rc in (merged.ranked_chunks or [])}
                scored = []
                for p in merged.evidence_packets:
                    cid = str(p.get("chunk_id") or "")
                    base = float(base_by_id.get(cid, 0.0))
                    rr = float(self.reranker.score(inputs.query, str(p.get("text_ar") or "")))
                    combo = (1.0 - alpha) * base + alpha * rr
                    scored.append((combo, rr, cid, p))
                scored.sort(key=lambda x: x[0], reverse=True)
                merged.evidence_packets = [x[3] for x in scored]
                # Rebuild ranked_chunks deterministically from reranked list.
                new_ranked = []
                for i, (combo, rr, cid, p) in enumerate(scored, start=1):
                    new_ranked.append(
                        {
                            "rank": i,
                            "chunk_id": cid,
                            "score": float(combo),
                            "reranker_score": float(rr),
                            "sources": sources_by_id.get(cid, []),
                            "backend": backend_by_id.get(cid, str(p.get("backend") or "")),
                        }
                    )
                merged.ranked_chunks = new_ranked
        except Exception:
            pass
        # For eval/observability: store last merge result on the instance.
        # Reason: runner/scorers need deterministic retrieval traces without changing
        # the public middleware API.
        try:
            self.last_merge_result = merged  # type: ignore[attr-defined]
        except Exception:
            pass
        return merged


