from __future__ import annotations


def test_partial_scenario_dedupes_identical_packet_text() -> None:
    from apps.api.core.scholar_reasoning_compose_scenario import compose_partial_scenario_answer

    packets = [
        {"chunk_type": "evidence", "text_ar": "نص تجريبي مكرر", "chunk_id": "CH_1", "source_anchor": "a", "refs": []},
        {"chunk_type": "evidence", "text_ar": "نص تجريبي مكرر", "chunk_id": "CH_2", "source_anchor": "b", "refs": []},
        {"chunk_type": "evidence", "text_ar": "نص تجريبي مكرر", "chunk_id": "CH_3", "source_anchor": "c", "refs": []},
    ]
    ans, _, _ = compose_partial_scenario_answer(packets=packets, question_ar="اختبار", prefer_more_claims=True)
    # The quoted line itself must appear once (synthesis may echo an excerpt).
    assert ans.count("- نص تجريبي مكرر") == 1


def test_deep_answer_dedupes_identical_evidence_lines() -> None:
    from apps.api.core.scholar_reasoning_compose import compose_deep_answer

    packets = [
        {"chunk_type": "definition", "text_ar": "تعريف", "chunk_id": "CH_def", "source_anchor": "a", "refs": []},
        {"chunk_type": "evidence", "text_ar": "دليل مكرر", "chunk_id": "CH_e1", "source_anchor": "b", "refs": []},
        {"chunk_type": "evidence", "text_ar": "دليل مكرر", "chunk_id": "CH_e2", "source_anchor": "c", "refs": []},
        {"chunk_type": "evidence", "text_ar": "دليل مكرر", "chunk_id": "CH_e3", "source_anchor": "d", "refs": []},
    ]
    ans, _, _ = compose_deep_answer(packets=packets, semantic_edges=[], max_edges=0, question_ar="اختبار", prefer_more_claims=True)
    assert ans.count("- دليل مكرر") == 1

