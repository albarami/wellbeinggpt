from __future__ import annotations


def test_build_argument_chains_from_used_edges_detects_boundary_and_inference_type() -> None:
    from apps.api.core.answer_contract import UsedEdge, UsedEdgeSpan, build_argument_chains_from_used_edges

    ue = UsedEdge(
        edge_id="E1",
        from_node="pillar:P001",
        to_node="pillar:P004",
        relation_type="ENABLES",
        justification_spans=(
            UsedEdgeSpan(chunk_id="CH_A", span_start=0, span_end=10, quote="فالجسم السليم هو وسيلة ..."),
            UsedEdgeSpan(chunk_id="CH_A", span_start=11, span_end=30, quote="ومن ضوابط ذلك: ..."),
        ),
    )
    chains = build_argument_chains_from_used_edges(used_edges=[ue])
    assert len(chains) == 1
    c = chains[0]
    assert c.edge_id == "E1"
    assert c.inference_type == "direct_quote"  # one non-boundary evidence span
    assert c.boundary_ar != "غير منصوص عليه في الإطار"
    assert c.boundary_spans

