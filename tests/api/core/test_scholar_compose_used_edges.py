from __future__ import annotations


def test_compose_deep_answer_emits_used_edges() -> None:
    from apps.api.core.scholar_reasoning_compose import compose_deep_answer

    packets = [
        {"chunk_type": "definition", "text_ar": "تعريف 1", "chunk_id": "CH_def", "source_anchor": "a", "refs": []},
        {"chunk_type": "evidence", "text_ar": "دليل 1", "chunk_id": "CH_ev", "source_anchor": "b", "refs": []},
        {"chunk_type": "commentary", "text_ar": "تطبيق 1", "chunk_id": "CH_com", "source_anchor": "c", "refs": []},
    ]
    semantic_edges = [
        {
            "edge_id": "E123",
            "relation_type": "ENABLES",
            "source_type": "pillar",
            "source_id": "P004",
            "neighbor_type": "pillar",
            "neighbor_id": "P001",
            "direction": "outgoing",
            "justification_spans": [
                {"chunk_id": "CH_j", "span_start": 0, "span_end": 10, "quote": "شاهد"},
            ],
        }
    ]

    ans, citations, used_edges = compose_deep_answer(
        packets=packets,
        semantic_edges=semantic_edges,
        max_edges=4,
        question_ar="اختبار",
        prefer_more_claims=False,
    )
    assert "الربط بين الركائز" in ans
    assert citations
    assert used_edges and used_edges[0]["edge_id"] == "E123"
    assert used_edges[0]["from_node"] == "pillar:P004"
    assert used_edges[0]["to_node"] == "pillar:P001"
    assert used_edges[0]["relation_type"] == "ENABLES"
    assert used_edges[0]["justification_spans"][0]["chunk_id"] == "CH_j"

