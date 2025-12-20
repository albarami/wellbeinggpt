import pytest

from apps.api.llm.gpt5_client_azure import MockProvider
from apps.api.llm.muhasibi_llm_client import MuhasibiLLMClient


@pytest.mark.asyncio
async def test_interpret_uses_mode_specific_prompt():
    provider = MockProvider(
        default_json={
            "answer_ar": "نص",
            "citations": [{"chunk_id": "CH_1", "source_anchor": "a", "ref": None}],
            "entities": [{"type": "core_value", "id": "CV1", "name_ar": "الإيمان"}],
            "not_found": False,
            "confidence": "high",
        }
    )
    client = MuhasibiLLMClient(provider)

    await client.interpret("سؤال", [{"chunk_id": "CH_1", "text_ar": "x", "source_doc_id": "d", "source_anchor": "a", "refs": []}], [], mode="debate")
    await client.interpret("سؤال", [{"chunk_id": "CH_1", "text_ar": "x", "source_doc_id": "d", "source_anchor": "a", "refs": []}], [], mode="socratic")
    await client.interpret("سؤال", [{"chunk_id": "CH_1", "text_ar": "x", "source_doc_id": "d", "source_anchor": "a", "refs": []}], [], mode="judge")
    await client.interpret("سؤال", [{"chunk_id": "CH_1", "text_ar": "x", "source_doc_id": "d", "source_anchor": "a", "refs": []}], [], mode="natural_chat")

    assert "Debate Mode" in provider.requests[0].system_prompt
    assert "Socratic Mode" in provider.requests[1].system_prompt
    assert "Judge Mode" in provider.requests[2].system_prompt
    # The natural chat prompt title is bilingual and does not include the word "Mode".
    assert "Natural Scholar Chat" in provider.requests[3].system_prompt


