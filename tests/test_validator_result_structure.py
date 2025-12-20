"""
Validator tests (result + structure + pillar).

Reason: keep each test file <500 LOC (project rule).
"""

from apps.api.ingest.validator import (
    Validator,
    ValidationIssue,
    ValidationResult,
    ValidationSeverity,
)
from apps.api.ingest.rule_extractor import ExtractionResult, ExtractedPillar


class TestValidationResult:
    """Tests for ValidationResult dataclass."""

    def test_counts_computed(self):
        """Test that issue counts are computed correctly."""
        issues = [
            ValidationIssue(severity=ValidationSeverity.ERROR, code="E1", message="Error 1"),
            ValidationIssue(severity=ValidationSeverity.ERROR, code="E2", message="Error 2"),
            ValidationIssue(severity=ValidationSeverity.WARNING, code="W1", message="Warning 1"),
            ValidationIssue(severity=ValidationSeverity.INFO, code="I1", message="Info 1"),
        ]

        result = ValidationResult(is_valid=False, issues=issues)
        assert result.error_count == 2
        assert result.warning_count == 1
        assert result.info_count == 1


class TestValidatorStructure:
    """Tests for overall structure validation."""

    def test_empty_document_fails(self):
        """Test that empty document produces error."""
        result = ExtractionResult(
            source_doc_id="DOC_test",
            source_file_hash="test",
            source_doc="docs/source/framework_2025-10_v1.docx",
            framework_version="2025-10",
            pillars=[],
        )

        validator = Validator()
        validation = validator.validate(result)

        assert not validation.is_valid
        assert any(i.code == "EMPTY_DOCUMENT" for i in validation.issues)

    def test_minimum_pillars_enforced(self):
        """Test that minimum pillar count is enforced."""
        result = ExtractionResult(
            source_doc_id="DOC_test",
            source_file_hash="test",
            source_doc="docs/source/framework_2025-10_v1.docx",
            framework_version="2025-10",
            pillars=[],
        )

        validator = Validator(min_pillars=5)
        validation = validator.validate(result)

        assert not validation.is_valid
        assert any(i.code == "MIN_PILLARS" for i in validation.issues)


class TestValidatorPillar:
    """Tests for pillar validation."""

    def test_missing_pillar_name_fails(self):
        """Test that pillar without name produces error."""
        pillar = ExtractedPillar(
            id="P001",
            name_ar="",  # Empty name
            source_doc="docs/source/framework_2025-10_v1.docx",
            source_hash="test",
            source_anchor="para_0",
            raw_text="",
            para_index=0,
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
        assert any(i.code == "MISSING_NAME" and i.entity_type == "pillar" for i in validation.issues)

    def test_pillar_without_core_values_warning(self):
        """Test that pillar without core values produces warning."""
        pillar = ExtractedPillar(
            id="P001",
            name_ar="الحياة الروحية",
            source_doc="docs/source/framework_2025-10_v1.docx",
            source_hash="test",
            source_anchor="para_0",
            raw_text="الحياة الروحية",
            para_index=0,
            core_values=[],
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

        assert validation.is_valid
        assert any(i.code == "NO_CORE_VALUES" for i in validation.issues)

