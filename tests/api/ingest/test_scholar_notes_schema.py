"""Scholar notes schema tests (pure validation).

These tests are DB-independent and ensure our JSONL schema contract is stable.
"""

from __future__ import annotations

import pytest

from apps.api.ingest.scholar_notes_schema import ScholarNoteRow, ScholarRelationType


def test_schema_validates_minimal_row() -> None:
    row = ScholarNoteRow.model_validate(
        {
            "note_id": "sn-test-001",
            "pillar_id": "P001",
            "core_value_id": None,
            "sub_value_id": None,
            "title_ar": "ملاحظة اختبار",
            "definition_ar": "",
            "deep_explanation_ar": "",
            "cross_pillar_links": [],
            "applied_scenarios": [],
            "common_misunderstandings": [],
            "evidence_spans": [
                {
                    "source_id": "DOC_ID",
                    "chunk_id": "CH_ABC",
                    "span_start": 0,
                    "span_end": 3,
                    "quote": "نص",
                }
            ],
            "tags": ["اختبار"],
            "version": "v1",
        }
    )
    assert row.note_id == "sn-test-001"


def test_schema_rejects_bad_relation_type() -> None:
    with pytest.raises(Exception):
        ScholarNoteRow.model_validate(
            {
                "note_id": "sn-test-002",
                "pillar_id": "P001",
                "core_value_id": None,
                "sub_value_id": None,
                "title_ar": "ملاحظة اختبار",
                "definition_ar": "تعريف",
                "deep_explanation_ar": "شرح",
                "cross_pillar_links": [
                    {
                        "target_pillar_id": "P002",
                        "target_sub_value_id": None,
                        "relation_type": "NOT_ALLOWED",
                        "justification_ar": "تبرير",
                    }
                ],
                "applied_scenarios": [],
                "common_misunderstandings": [],
                "evidence_spans": [
                    {
                        "source_id": "DOC_ID",
                        "chunk_id": "CH_ABC",
                        "span_start": 0,
                        "span_end": 3,
                        "quote": "نص",
                    }
                ],
                "tags": [],
                "version": "v1",
            }
        )


def test_relation_type_enum_values_stable() -> None:
    assert ScholarRelationType.COMPLEMENTS.value == "COMPLEMENTS"
    assert ScholarRelationType.RESOLVES_WITH.value == "RESOLVES_WITH"
