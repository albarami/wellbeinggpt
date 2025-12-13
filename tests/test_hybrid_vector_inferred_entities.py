import pytest

from apps.api.retrieve.hybrid_retriever import HybridRetriever, RetrievalInputs
from apps.api.core.schemas import EntityType


@pytest.mark.asyncio
async def test_hybrid_retriever_infers_entities_from_vector_results(monkeypatch):
    """
    Expected: when no entities resolved, vector hits infer entity ids and trigger graph expansion.
    """

    class DummyVector:
        async def search(self, session, query, top_k=10, entity_types=None, chunk_types=None):
            return [
                {
                    "chunk_id": "CH_1",
                    "entity_type": "sub_value",
                    "entity_id": "SV_X",
                    "chunk_type": "definition",
                    "text_ar": "نص",
                    "source_doc_id": "DOC",
                    "source_anchor": "a",
                    "refs": [{"type": "quran", "ref": "البقرة:1"}],
                }
            ]

    calls = {"expand": 0}

    async def fake_expand_graph(session, entity_type, entity_id, depth=2, relationship_types=None):
        calls["expand"] += 1
        assert entity_type == EntityType.SUB_VALUE
        assert entity_id == "SV_X"
        return [{"neighbor_type": "sub_value", "neighbor_id": "SV_Y", "depth": 1}]

    async def fake_get_chunks_with_refs(session, et, eid, limit=20):
        return [
            {
                "chunk_id": "CH_2",
                "entity_type": et.value,
                "entity_id": eid,
                "chunk_type": "definition",
                "text_ar": "نص",
                "source_doc_id": "DOC",
                "source_anchor": "b",
                "refs": [],
            }
        ]

    retriever = HybridRetriever(enable_vector=True, enable_graph=True)
    retriever.vector_retriever = DummyVector()

    async def no_ref_expand(self, session, ref_node_ids, depth=2, per_entity_limit=6):
        return []

    monkeypatch.setattr(HybridRetriever, "_expand_via_refs", no_ref_expand)
    monkeypatch.setattr("apps.api.retrieve.hybrid_retriever.expand_graph", fake_expand_graph)
    monkeypatch.setattr("apps.api.retrieve.hybrid_retriever.get_chunks_with_refs", fake_get_chunks_with_refs)

    res = await retriever.retrieve(
        session=None,  # mocked away
        inputs=RetrievalInputs(query="الإنتاجية", resolved_entities=[], top_k=5, graph_depth=2),
    )

    assert calls["expand"] >= 1
    assert res.total_found >= 1


