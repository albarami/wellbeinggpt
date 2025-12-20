from __future__ import annotations


def test_framework_edge_miner_expected_enables_edges() -> None:
    """
    Expected: a physical paragraph explicitly enabling multiple pillars yields ENABLES edges.
    """

    from apps.api.graph.framework_edge_miner import extract_semantic_edges_from_chunk

    txt = (
        "إن سلامة الجسد وسيلة لإطلاق الطاقات الروحية والعاطفية والعقلية والاجتماعية. "
        "ولا يفصل الإسلام بين الجسد والروح.\n"
    )
    mined = extract_semantic_edges_from_chunk(chunk_id="CH_TEST", text_ar=txt)
    # Must at least produce the 4 ENABLES edges from Physical -> {P001,P002,P003,P005}
    keys = {(m.from_pillar_id, m.to_pillar_id, m.relation_type) for m in mined}
    assert ("P004", "P001", "ENABLES") in keys
    assert ("P004", "P002", "ENABLES") in keys
    assert ("P004", "P003", "ENABLES") in keys
    assert ("P004", "P005", "ENABLES") in keys


def test_framework_edge_miner_integration_produces_complements_bidirectional() -> None:
    """
    Expected: explicit integration language across pillars produces bidirectional COMPLEMENTS.
    """

    from apps.api.graph.framework_edge_miner import extract_semantic_edges_from_chunk

    txt = "هذا الإطار منظومة متكاملة لا ينفصل فيها البعد البدني عن البعد الروحي.\n"
    mined = extract_semantic_edges_from_chunk(chunk_id="CH_TEST", text_ar=txt)
    keys = {(m.from_pillar_id, m.to_pillar_id, m.relation_type) for m in mined}
    assert ("P004", "P001", "COMPLEMENTS") in keys
    assert ("P001", "P004", "COMPLEMENTS") in keys


def test_framework_edge_miner_conditional_produces_conditional_on() -> None:
    """
    Expected: 'لا يتحقق X إلا ب Y' yields X CONDITIONAL_ON Y.
    """

    from apps.api.graph.framework_edge_miner import extract_semantic_edges_from_chunk

    txt = "لا تتحقق الحياة الروحية إلا بسلامة الجسد.\n"
    mined = extract_semantic_edges_from_chunk(chunk_id="CH_TEST", text_ar=txt)
    keys = {(m.from_pillar_id, m.to_pillar_id, m.relation_type) for m in mined}
    assert ("P001", "P004", "CONDITIONAL_ON") in keys


def test_framework_edge_miner_edge_case_no_marker_no_edges() -> None:
    """
    Edge case: mentions multiple pillars but no explicit relation marker -> emit nothing.
    """

    from apps.api.graph.framework_edge_miner import extract_semantic_edges_from_chunk

    txt = "الحياة البدنية والحياة الروحية تذكران هنا دون بيان علاقة.\n"
    mined = extract_semantic_edges_from_chunk(chunk_id="CH_TEST", text_ar=txt)
    assert mined == []


def test_framework_edge_miner_failure_empty_text_safe() -> None:
    """
    Failure case: empty text should not crash and returns no edges.
    """

    from apps.api.graph.framework_edge_miner import extract_semantic_edges_from_chunk

    mined = extract_semantic_edges_from_chunk(chunk_id="CH_TEST", text_ar="")
    assert mined == []

