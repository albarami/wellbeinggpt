from __future__ import annotations

import pytest

from eval.scoring.graph import score_graph
from eval.types import EvalCitation, EvalMode, EvalOutputRow, GraphTrace, GraphTracePath, RetrievalTrace


@pytest.mark.asyncio
async def test_cross_pillar_grounding_requires_justification():
    # Dataset expects a justification token.
    dataset_by_id = {
        "q1": {
            "id": "q1",
            "type": "cross_pillar",
            "required_graph_paths": [
                {"nodes": ["pillar:a", "pillar:b"], "edges": ["pillar:a::REL::pillar:b"], "justification": "quran:[2:1]"}
            ],
        }
    }

    ok_row = EvalOutputRow(
        id="q1",
        mode=EvalMode.RAG_PLUS_GRAPH,
        question="سؤال",
        answer_ar="جملة واحدة.",
        claims=[],
        citations=[EvalCitation(source_id="dummy", span_start=0, span_end=1, quote="... quran:[2:1] ...")],
        retrieval_trace=RetrievalTrace(),
        graph_trace=GraphTrace(nodes=["pillar:a", "pillar:b"], edges=["pillar:a::REL::pillar:b"], paths=[GraphTracePath(nodes=["pillar:a","pillar:b"], edges=["pillar:a::REL::pillar:b"], confidence=1.0)]),
        abstained=False,
        abstain_reason=None,
        latency_ms=1,
        debug={},
    )

    bad_row = ok_row.model_copy(update={"citations": [EvalCitation(source_id="dummy", span_start=0, span_end=1, quote="no match") ]})

    m_ok = await score_graph(session=None, outputs=[ok_row], dataset_by_id=dataset_by_id)
    m_bad = await score_graph(session=None, outputs=[bad_row], dataset_by_id=dataset_by_id)

    assert m_ok.explanation_grounded_rate == 1.0
    assert m_bad.explanation_grounded_rate == 0.0


@pytest.mark.asyncio
async def test_cross_pillar_shuffle_citations_drops_grounded_rate():
    dataset_by_id = {
        "a": {
            "id": "a",
            "type": "cross_pillar",
            "required_graph_paths": [{"nodes": ["pillar:x", "pillar:y"], "edges": [], "justification": "REF_A"}],
        },
        "b": {
            "id": "b",
            "type": "cross_pillar",
            "required_graph_paths": [{"nodes": ["pillar:x", "pillar:z"], "edges": [], "justification": "REF_B"}],
        },
    }

    row_a = EvalOutputRow(
        id="a",
        mode=EvalMode.RAG_PLUS_GRAPH,
        question="سؤال A",
        answer_ar="...",
        claims=[],
        citations=[EvalCitation(source_id="dummy", span_start=0, span_end=1, quote="... REF_A ...")],
        retrieval_trace=RetrievalTrace(),
        graph_trace=GraphTrace(edges=["e"], paths=[GraphTracePath(nodes=[], edges=["e"], confidence=1.0)]),
        abstained=False,
        abstain_reason=None,
        latency_ms=1,
        debug={},
    )
    row_b = row_a.model_copy(update={"id": "b", "question": "سؤال B", "citations": [EvalCitation(source_id="dummy", span_start=0, span_end=1, quote="... REF_B ...") ]})

    # Shuffle citations
    row_a_shuf = row_a.model_copy(update={"citations": row_b.citations})
    row_b_shuf = row_b.model_copy(update={"citations": row_a.citations})

    m_before = await score_graph(session=None, outputs=[row_a, row_b], dataset_by_id=dataset_by_id)
    m_after = await score_graph(session=None, outputs=[row_a_shuf, row_b_shuf], dataset_by_id=dataset_by_id)

    assert m_before.explanation_grounded_rate == 1.0
    assert m_after.explanation_grounded_rate == 0.0
