from __future__ import annotations

from apps.api.core.answer_contract import (
    UsedEdge,
    UsedEdgeSpan,
    check_contract,
    contract_from_answer_requirements,
    extract_compare_concepts_from_question,
)


def test_extract_compare_concepts_from_question() -> None:
    q = "عرّف التزكية داخل هذا الإطار، ثم بيّن الفرق بينها وبين العبادة والإيمان من حيث التعريف."
    concepts = extract_compare_concepts_from_question(q)
    assert "العبادة" in concepts
    assert "الإيمان" in concepts


def test_contract_from_answer_requirements_compare_infers_required_entities() -> None:
    spec = contract_from_answer_requirements(
        question_norm="",
        question_ar="عرّف التزكية ثم بيّن الفرق بينها وبين العبادة والإيمان.",
        question_type="compare",
        answer_requirements={"format": "scholar", "must_include": ["تعريف المفهوم داخل الإطار", "خلاصة تنفيذية"]},
    )
    assert spec.intent_type == "compare"
    assert "مصفوفة المقارنة" in spec.required_sections
    assert "العبادة" in spec.required_entities
    assert "الإيمان" in spec.required_entities


def test_check_contract_requires_graph_edges_when_requested() -> None:
    spec = contract_from_answer_requirements(
        question_norm="",
        question_ar="اربط بين الركائز مع مسار.",
        question_type="cross_pillar",
        answer_requirements={"format": "scholar", "must_include": ["الربط بين الركائز", "path_trace", "citations"]},
    )
    cm = check_contract(spec=spec, answer_ar="الربط بين الركائز (مع سبب الربط)\n- ...", citations=[{"x": 1}], used_edges=[])
    assert cm.outcome.value == "FAIL"
    assert "MISSING_USED_GRAPH_EDGES" in cm.reasons


def test_check_contract_compare_requires_fields_per_concept() -> None:
    spec = contract_from_answer_requirements(
        question_norm="",
        question_ar="بيّن الفرق بينها وبين العبادة والإيمان.",
        question_type="compare",
        answer_requirements={"format": "scholar", "must_include": ["تعريف المفهوم داخل الإطار"]},
    )
    answer = "\n".join(
        [
            "مصفوفة المقارنة",
            "- العبادة:",
            "- التعريف: غير منصوص عليه",
            "- المظهر العملي: غير منصوص عليه",
            "- الخطأ الشائع: غير منصوص عليه",
            "خلاصة تنفيذية (3 نقاط)",
            "- x",
            "- y",
            "- z",
        ]
    )
    cm = check_contract(spec=spec, answer_ar=answer, citations=[{"x": 1}], used_edges=[])
    assert cm.outcome.value == "FAIL"
    assert any("COMPARE_MISSING_CONCEPT_BLOCK" in r for r in cm.reasons) or any(
        "COMPARE_MISSING_FIELD" in r for r in cm.reasons
    )


def test_check_contract_graph_distinct_pillars() -> None:
    spec = contract_from_answer_requirements(
        question_norm="",
        question_ar="ابن شبكة تربطها بثلاث ركائز أخرى.",
        question_type="cross_pillar",
        answer_requirements={"format": "scholar", "must_include": ["path_trace"]},
    )
    # Force "network" semantics via question_type, to keep this deterministic.
    spec = spec.__class__(
        intent_type="network",
        required_sections=spec.required_sections,
        required_entities=spec.required_entities,
        requires_graph=True,
        min_links=3,
        min_distinct_pillars=4,
        allow_followup_questions=False,
    )
    used = [
        UsedEdge(
            edge_id="e1",
            from_node="pillar:a",
            to_node="pillar:b",
            relation_type="ENABLES",
            justification_spans=(UsedEdgeSpan(chunk_id="CH_x", span_start=0, span_end=5, quote="q"),),
        ),
        UsedEdge(
            edge_id="e2",
            from_node="pillar:a",
            to_node="pillar:c",
            relation_type="ENABLES",
            justification_spans=(UsedEdgeSpan(chunk_id="CH_x", span_start=0, span_end=5, quote="q"),),
        ),
        UsedEdge(
            edge_id="e3",
            from_node="pillar:b",
            to_node="pillar:c",
            relation_type="ENABLES",
            justification_spans=(UsedEdgeSpan(chunk_id="CH_x", span_start=0, span_end=5, quote="q"),),
        ),
    ]
    cm = check_contract(spec=spec, answer_ar="x", citations=[{"x": 1}], used_edges=used)
    assert cm.outcome.value == "FAIL"
    assert "INSUFFICIENT_DISTINCT_PILLARS_IN_GRAPH" in cm.reasons


def test_contract_from_answer_requirements_scenario_uses_naturalized_markers() -> None:
    spec = contract_from_answer_requirements(
        question_norm="",
        question_ar="حلّل الحالة وفق الإطار.",
        question_type="scenario",
        answer_requirements={"format": "scholar", "must_include": ["citations"]},
    )
    assert spec.intent_type == "scenario"
    assert "ما يمكن دعمه من الأدلة المسترجعة" in spec.required_sections
    assert "ما لا يمكن الجزم به من الأدلة الحالية" in spec.required_sections

