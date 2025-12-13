import pytest

from apps.api.llm.gpt5_client_azure import MockProvider
from apps.api.llm.muhasibi_llm_client import MuhasibiLLMClient


@pytest.mark.asyncio
async def test_query_rewrite_ar_returns_rewrites():
    provider = MockProvider(
        default_json={
            "rewrites_ar": ["تعريف الإيمان", "معنى الإيمان في الإطار"],
            "focus_terms_ar": ["الإيمان"],
            "disambiguation_question_ar": None,
        }
    )
    client = MuhasibiLLMClient(provider)
    res = await client.query_rewrite_ar(
        question="ما هو الايمان؟",
        detected_entities=[{"type": "core_value", "name_ar": "الإيمان", "confidence": 0.9}],
        keywords=["ايمان"],
    )
    assert res is not None
    assert "rewrites_ar" in res and len(res["rewrites_ar"]) >= 1


