import pytest

from apps.api.core.muhasibi_state_machine import create_middleware


@pytest.mark.asyncio
async def test_process_with_trace_returns_state_flow():
    middleware = create_middleware()
    final, trace = await middleware.process_with_trace("ما هو الإيمان؟", language="ar", mode="answer")

    assert final is not None
    assert isinstance(trace, list)
    states = [t.get("state") for t in trace]
    # At minimum we should see the pipeline states in order
    assert "LISTEN" in states
    assert "RETRIEVE" in states
    assert "ACCOUNT" in states
    assert "INTERPRET" in states

    # Ensure trace is high-level (no evidence packet text dumped)
    for t in trace:
        assert "evidence_packets" not in t


