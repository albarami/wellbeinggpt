"""
Hybrid Retrieval Pipeline

Orchestrates:
- Entity-first SQL retrieval
- Graph expansion (edges table)
- Vector retrieval (pgvector) when embeddings are available

Returns Evidence Packets (contract) via MergeRanker.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from sqlalchemy.ext.asyncio import AsyncSession

from apps.api.core.schemas import EntityType
from apps.api.retrieve.merge_rank import MergeRanker, MergeResult
from apps.api.retrieve.sql_retriever import get_chunks_with_refs
from apps.api.retrieve.graph_retriever import expand_graph
from apps.api.retrieve.vector_retriever import VectorRetriever


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
        enable_vector: bool = True,
        enable_graph: bool = True,
    ):
        self.merge_ranker = merge_ranker or MergeRanker(max_packets=10)
        self.vector_retriever = vector_retriever or VectorRetriever()
        self.enable_vector = enable_vector
        self.enable_graph = enable_graph

    async def retrieve(
        self,
        session: AsyncSession,
        inputs: RetrievalInputs,
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

        return self.merge_ranker.merge(
            sql_results=sql_results,
            vector_results=vector_results,
            graph_results=graph_results,
            resolved_entities=inputs.resolved_entities,
        )


