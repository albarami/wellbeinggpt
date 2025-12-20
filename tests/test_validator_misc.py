"""
Validator tests (duplicates, sub-values, evidence refs, convenience wrapper).

Reason: keep each test file <500 LOC (project rule).
"""

from apps.api.ingest.validator import (
    Validator,
    ValidationSeverity,
    validate_evidence_refs,
    validate_extraction,
)
from apps.api.ingest.rule_extractor import (
    ExtractionResult,
    ExtractedCoreValue,
    ExtractedDefinition,
    ExtractedPillar,
    ExtractedSubValue,
)
from apps.api.ingest.evidence_parser import ParsedQuranRef, ParseStatus


class TestValidatorDuplicates:
    """Tests for duplicate name detection."""

    def test_duplicate_pillar_names_fail(self):
        """Test that duplicate pillar names produce error."""
        pillar1 = ExtractedPillar(
            id="P001",
            name_ar="الحياة الروحية",
            source_doc="docs/source/framework_2025-10_v1.docx",
            source_hash="test",
            source_anchor="para_0",
            raw_text="الحياة الروحية",
            para_index=0,
        )
        pillar2 = ExtractedPillar(
            id="P002",
            name_ar="الحياة الروحية",  # Duplicate
            source_doc="docs/source/framework_2025-10_v1.docx",
            source_hash="test",
            source_anchor="para_10",
            raw_text="الحياة الروحية",
            para_index=10,
        )
        result = ExtractionResult(
            source_doc_id="DOC_test",
            source_file_hash="test",
            source_doc="docs/source/framework_2025-10_v1.docx",
            framework_version="2025-10",
            pillars=[pillar1, pillar2],
        )

        validator = Validator(allow_duplicate_names=False)
        validation = validator.validate(result)

        assert not validation.is_valid
        assert any(i.code == "DUPLICATE_NAME" for i in validation.issues)

    def test_duplicates_ok_when_allowed(self):
        """Test that duplicates are ok when allowed."""
        pillar1 = ExtractedPillar(
            id="P001",
            name_ar="الحياة الروحية",
            source_doc="docs/source/framework_2025-10_v1.docx",
            source_hash="test",
            source_anchor="para_0",
            raw_text="الحياة الروحية",
            para_index=0,
        )
        pillar2 = ExtractedPillar(
            id="P002",
            name_ar="الحياة الروحية",
            source_doc="docs/source/framework_2025-10_v1.docx",
            source_hash="test",
            source_anchor="para_10",
            raw_text="الحياة الروحية",
            para_index=10,
        )
        result = ExtractionResult(
            source_doc_id="DOC_test",
            source_file_hash="test",
            source_doc="docs/source/framework_2025-10_v1.docx",
            framework_version="2025-10",
            pillars=[pillar1, pillar2],
        )

        validator = Validator(allow_duplicate_names=True)
        validation = validator.validate(result)
        assert not any(i.code == "DUPLICATE_NAME" for i in validation.issues)


class TestValidatorSubValue:
    """Tests for sub-value validation."""

    def test_missing_sub_value_name_fails(self):
        """Test that sub-value without name produces error."""
        sv = ExtractedSubValue(
            id="SV001",
            name_ar="",  # Empty
            source_doc="docs/source/framework_2025-10_v1.docx",
            source_hash="test",
            source_anchor="para_2",
            raw_text="",
            para_index=2,
        )
        cv = ExtractedCoreValue(
            id="CV001",
            name_ar="الإيمان",
            source_doc="docs/source/framework_2025-10_v1.docx",
            source_hash="test",
            source_anchor="para_1",
            raw_text="الإيمان",
            para_index=1,
            definition=ExtractedDefinition(
                text_ar="تعريف",
                source_doc="docs/source/framework_2025-10_v1.docx",
                source_hash="test",
                source_anchor="para_3",
                raw_text="تعريف",
            ),
            sub_values=[sv],
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

        assert not validation.is_valid
        assert any(i.code == "MISSING_NAME" and i.entity_type == "sub_value" for i in validation.issues)


class TestValidateEvidenceRefs:
    """Tests for evidence reference validation."""

    def test_failed_parse_produces_error(self):
        """Test that failed parse status produces error."""
        refs = [
            ParsedQuranRef(
                surah_name_ar="Unknown",
                ayah_number=1,
                ref_raw="[???]",
                ref_norm="???:1",
                parse_status=ParseStatus.FAILED,
            )
        ]

        issues = validate_evidence_refs(refs)
        assert len(issues) == 1
        assert issues[0].severity == ValidationSeverity.ERROR
        assert issues[0].code == "PARSE_FAILED"

    def test_needs_review_produces_warning(self):
        """Test that needs_review status produces warning."""
        refs = [
            ParsedQuranRef(
                surah_name_ar="Unknown",
                ayah_number=1,
                ref_raw="[something]",
                ref_norm="something:1",
                parse_status=ParseStatus.NEEDS_REVIEW,
            )
        ]

        issues = validate_evidence_refs(refs)
        assert len(issues) == 1
        assert issues[0].severity == ValidationSeverity.WARNING
        assert issues[0].code == "NEEDS_REVIEW"

    def test_success_produces_no_issues(self):
        """Test that success status produces no issues."""
        refs = [
            ParsedQuranRef(
                surah_name_ar="البقرة",
                surah_number=2,
                ayah_number=1,
                ref_raw="[البقرة: 1]",
                ref_norm="البقرة:1",
                parse_status=ParseStatus.SUCCESS,
            )
        ]

        issues = validate_evidence_refs(refs)
        assert len(issues) == 0


class TestValidateExtractionConvenience:
    """Tests for the convenience function."""

    def test_strict_mode(self):
        """Test strict mode requires definitions."""
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

        validation = validate_extraction(result, strict=True)
        assert not validation.is_valid

        validation = validate_extraction(result, strict=False)
        assert validation.is_valid

