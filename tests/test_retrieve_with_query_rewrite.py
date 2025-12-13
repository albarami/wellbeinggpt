import pytest
from unittest.mock import AsyncMock

from apps.api.core.muhasibi_state_machine import create_middleware, StateContext
from apps.api.llm.gpt5_client_azure import MockProvider
from apps.api.llm.muhasibi_llm_client import MuhasibiLLMClient
from apps.api.retrieve.hybrid_retriever import HybridRetriever, RetrievalInputs


@pytest.mark.asyncio
async def test_retrieve_uses_llm_query_rewrite_when_empty(monkeypatch):
    """
    Expected: when initial retrieval returns empty and llm_client is configured,
    middleware performs extra retrieval attempts using rewrites.
    """
    provider = MockProvider(
        default_json={
            "rewrites_ar": ["تعريف الإيمان"],
            "focus_terms_ar": ["الإيمان"],
            "disambiguation_question_ar": None,
        }
    )
    llm_client = MuhasibiLLMClient(provider)

    retriever = HybridRetriever(enable_vector=False, enable_graph=False)
    retriever._session = object()  # sentinel to satisfy middleware path

    async def fake_retrieve(session, inputs: RetrievalInputs):
        # First call (original question) returns empty; rewrite returns 1 packet
        if inputs.query == "ما هو الإيمان؟":
            return type("R", (), {"evidence_packets": [], "has_definition": False, "has_evidence": False})()
        return type(
            "R",
            (),
            {
                "evidence_packets": [
                    {
                        "chunk_id": "CH_X",
                        "entity_type": "core_value",
                        "entity_id": "CV001",
                        "chunk_type": "definition",
                        "text_ar": "تعريف",
                        "source_doc_id": "DOC",
                        "source_anchor": "a",
                        "refs": [],
                    }
                ],
                "has_definition": True,
                "has_evidence": False,
            },
        )()

    monkeypatch.setattr(retriever, "retrieve", fake_retrieve)

    middleware = create_middleware(retriever=retriever, llm_client=llm_client)
    ctx = StateContext(question="ما هو الإيمان؟", language="ar")
    await middleware._state_listen(ctx)
    await middleware._state_retrieve(ctx)

    assert any(p.get("chunk_id") == "CH_X" for p in ctx.evidence_packets)


