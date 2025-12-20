"""
Validator tests (core value rules).

Reason: keep each test file <500 LOC (project rule).
"""

from apps.api.ingest.validator import Validator
from apps.api.ingest.rule_extractor import (
    ExtractionResult,
    ExtractedCoreValue,
    ExtractedDefinition,
    ExtractedPillar,
    ExtractedSubValue,
)


class TestValidatorCoreValue:
    """Tests for core value validation."""

    def test_missing_core_value_name_fails(self):
        """Test that core value without name produces error."""
        cv = ExtractedCoreValue(
            id="CV001",
            name_ar="",  # Empty
            source_doc="docs/source/framework_2025-10_v1.docx",
            source_hash="test",
            source_anchor="para_1",
            raw_text="",
            para_index=1,
        )
        pillar = ExtractedPillar(
            id="P001",
            name_ar="الحياة الروحية",
            source_doc="docs/source/framework_2025-10_v1.docx",
            source_hash="test",
            source_anchor="para_0",
            raw_text="الحياة الروحية",
            para_index=0,
            core_values=[cv],
        )
        result = ExtractionResult(
            source_doc_id="DOC_test",
            source_file_hash="test",
            source_doc="docs/source/framework_2025-10_v1.docx",
            framework_version="2025-10",
            pillars=[pillar],
        )

        validator = Validator()
        validation = validator.validate(result)

        assert not validation.is_valid
        assert any(i.code == "MISSING_NAME" and i.entity_type == "core_value" for i in validation.issues)

    def test_missing_definition_fails_when_required(self):
        """Test missing definition is error when required (and no sub-values exist)."""
        cv = ExtractedCoreValue(
            id="CV001",
            name_ar="الإيمان",
            source_doc="docs/source/framework_2025-10_v1.docx",
            source_hash="test",
            source_anchor="para_1",
            raw_text="الإيمان",
            para_index=1,
            definition=None,
        )
        pillar = ExtractedPillar(
            id="P001",
            name_ar="الحياة الروحية",
            source_doc="docs/source/framework_2025-10_v1.docx",
            source_hash="test",
            source_anchor="para_0",
            raw_text="الحياة الروحية",
            para_index=0,
            core_values=[cv],
        )
        result = ExtractionResult(
            source_doc_id="DOC_test",
            source_file_hash="test",
            source_doc="docs/source/framework_2025-10_v1.docx",
            framework_version="2025-10",
            pillars=[pillar],
        )

        validator = Validator(require_definitions=True)
        validation = validator.validate(result)

        assert not validation.is_valid
        assert any(i.code == "MISSING_DEFINITION" and i.entity_type == "core_value" for i in validation.issues)

    def test_missing_definition_ok_when_core_has_sub_values(self):
        """If core value is a container for sub-values, its definition can be absent."""
        sv = ExtractedSubValue(
            id="SV001",
            name_ar="التوحيد",
            source_doc="docs/source/framework_2025-10_v1.docx",
            source_hash="test",
            source_anchor="para_2",
            raw_text="التوحيد",
            para_index=2,
            definition=ExtractedDefinition(
                text_ar="تعريف التوحيد",
                source_doc="docs/source/framework_2025-10_v1.docx",
                source_hash="test",
                source_anchor="para_3",
                raw_text="تعريف التوحيد",
            ),
            evidence=[],
        )
        cv = ExtractedCoreValue(
            id="CV001",
            name_ar="التزكية",
            source_doc="docs/source/framework_2025-10_v1.docx",
            source_hash="test",
            source_anchor="para_1",
            raw_text="التزكية",
            para_index=1,
            definition=None,
            sub_values=[sv],
            evidence=[],
        )
        pillar = ExtractedPillar(
            id="P001",
            name_ar="الحياة الروحية",
            source_doc="docs/source/framework_2025-10_v1.docx",
            source_hash="test",
            source_anchor="para_0",
            raw_text="الحياة الروحية",
            para_index=0,
            core_values=[cv],
        )
        result = ExtractionResult(
            source_doc_id="DOC_test",
            source_file_hash="test",
            source_doc="docs/source/framework_2025-10_v1.docx",
            framework_version="2025-10",
            pillars=[pillar],
        )
        validator = Validator(require_definitions=True)
        validation = validator.validate(result)
        assert validation.is_valid

    def test_missing_definition_ok_when_not_required(self):
        """Test that missing definition is ok when not required."""
        cv = ExtractedCoreValue(
            id="CV001",
            name_ar="الإيمان",
            source_doc="docs/source/framework_2025-10_v1.docx",
            source_hash="test",
            source_anchor="para_1",
            raw_text="الإيمان",
            para_index=1,
            definition=None,
        )
        pillar = ExtractedPillar(
            id="P001",
            name_ar="الحياة الروحية",
            source_doc="docs/source/framework_2025-10_v1.docx",
            source_hash="test",
            source_anchor="para_0",
            raw_text="الحياة الروحية",
            para_index=0,
            core_values=[cv],
        )
        result = ExtractionResult(
            source_doc_id="DOC_test",
            source_file_hash="test",
            source_doc="docs/source/framework_2025-10_v1.docx",
            framework_version="2025-10",
            pillars=[pillar],
        )

        validator = Validator(require_definitions=False)
        validation = validator.validate(result)
        assert validation.is_valid

