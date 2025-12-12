"""
Validation Module

Implements validation gates for the ingestion pipeline.
Enforces "no hallucination" constraints by failing or flagging issues.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from apps.api.ingest.rule_extractor import (
    ExtractionResult,
    ExtractedPillar,
    ExtractedCoreValue,
    ExtractedSubValue,
)
from apps.api.ingest.evidence_parser import ParseStatus


class ValidationSeverity(str, Enum):
    """Severity levels for validation issues."""

    ERROR = "error"  # Fails ingestion
    WARNING = "warning"  # Flagged but continues
    INFO = "info"  # Informational


@dataclass
class ValidationIssue:
    """A validation issue found during ingestion."""

    severity: ValidationSeverity
    code: str
    message: str
    entity_type: Optional[str] = None
    entity_id: Optional[str] = None
    entity_name: Optional[str] = None
    field: Optional[str] = None
    para_index: Optional[int] = None


@dataclass
class ValidationResult:
    """Result of validation."""

    is_valid: bool
    issues: list[ValidationIssue] = field(default_factory=list)
    error_count: int = 0
    warning_count: int = 0
    info_count: int = 0

    def __post_init__(self):
        """Count issues by severity."""
        self.error_count = sum(
            1 for i in self.issues if i.severity == ValidationSeverity.ERROR
        )
        self.warning_count = sum(
            1 for i in self.issues if i.severity == ValidationSeverity.WARNING
        )
        self.info_count = sum(
            1 for i in self.issues if i.severity == ValidationSeverity.INFO
        )


class Validator:
    """
    Validator for extracted wellbeing framework content.

    Implements validation gates per the plan:
    - Missing value names → ERROR
    - Empty المفهوم sections → ERROR
    - Unparseable evidence references → WARNING (needs review)
    - Duplicate names under same parent → ERROR (unless allowed)
    """

    def __init__(
        self,
        allow_duplicate_names: bool = False,
        require_definitions: bool = True,
        require_evidence: bool = False,
        min_pillars: int = 1,
        max_definition_length: int = 5000,
    ):
        """
        Initialize the validator.

        Args:
            allow_duplicate_names: Allow duplicate names under same parent.
            require_definitions: Require definitions for values.
            require_evidence: Require at least one evidence per value.
            min_pillars: Minimum number of pillars required.
            max_definition_length: Maximum characters in a definition.
        """
        self.allow_duplicate_names = allow_duplicate_names
        self.require_definitions = require_definitions
        self.require_evidence = require_evidence
        self.min_pillars = min_pillars
        self.max_definition_length = max_definition_length

    def validate(self, result: ExtractionResult) -> ValidationResult:
        """
        Validate an extraction result.

        Args:
            result: The extraction result to validate.

        Returns:
            ValidationResult: Validation outcome with issues.
        """
        issues: list[ValidationIssue] = []

        # Validate overall structure
        issues.extend(self._validate_structure(result))

        # Validate each pillar
        for pillar in result.pillars:
            issues.extend(self._validate_pillar(pillar))

        # Check for duplicates across the whole document
        issues.extend(self._check_duplicates(result))

        # Determine if valid (no errors)
        is_valid = not any(
            i.severity == ValidationSeverity.ERROR for i in issues
        )

        return ValidationResult(is_valid=is_valid, issues=issues)

    def _validate_structure(
        self, result: ExtractionResult
    ) -> list[ValidationIssue]:
        """Validate overall document structure."""
        issues = []

        # Check minimum pillars
        if len(result.pillars) < self.min_pillars:
            issues.append(
                ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    code="MIN_PILLARS",
                    message=f"Expected at least {self.min_pillars} pillars, found {len(result.pillars)}",
                )
            )

        # Check for empty document
        if not result.pillars:
            issues.append(
                ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    code="EMPTY_DOCUMENT",
                    message="No pillars extracted from document",
                )
            )

        return issues

    def _validate_pillar(self, pillar: ExtractedPillar) -> list[ValidationIssue]:
        """Validate a pillar and its contents."""
        issues = []

        # Check pillar name
        if not pillar.name_ar or not pillar.name_ar.strip():
            issues.append(
                ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    code="MISSING_NAME",
                    message="Pillar is missing Arabic name",
                    entity_type="pillar",
                    entity_id=pillar.id,
                    field="name_ar",
                    para_index=pillar.para_index,
                )
            )

        # Check for core values
        if not pillar.core_values:
            issues.append(
                ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    code="NO_CORE_VALUES",
                    message=f"Pillar '{pillar.name_ar}' has no core values",
                    entity_type="pillar",
                    entity_id=pillar.id,
                    entity_name=pillar.name_ar,
                )
            )

        # Validate each core value
        for cv in pillar.core_values:
            issues.extend(self._validate_core_value(cv, pillar))

        return issues

    def _validate_core_value(
        self, cv: ExtractedCoreValue, pillar: ExtractedPillar
    ) -> list[ValidationIssue]:
        """Validate a core value and its contents."""
        issues = []

        # Check core value name
        if not cv.name_ar or not cv.name_ar.strip():
            issues.append(
                ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    code="MISSING_NAME",
                    message=f"Core value is missing Arabic name (under pillar '{pillar.name_ar}')",
                    entity_type="core_value",
                    entity_id=cv.id,
                    field="name_ar",
                    para_index=cv.para_index,
                )
            )

        # Check definition if required
        if self.require_definitions:
            if not cv.definition or not cv.definition.text_ar.strip():
                issues.append(
                    ValidationIssue(
                        severity=ValidationSeverity.ERROR,
                        code="MISSING_DEFINITION",
                        message=f"Core value '{cv.name_ar}' is missing definition (المفهوم)",
                        entity_type="core_value",
                        entity_id=cv.id,
                        entity_name=cv.name_ar,
                        field="definition",
                    )
                )
            elif len(cv.definition.text_ar) > self.max_definition_length:
                issues.append(
                    ValidationIssue(
                        severity=ValidationSeverity.WARNING,
                        code="DEFINITION_TOO_LONG",
                        message=f"Definition for '{cv.name_ar}' exceeds {self.max_definition_length} chars",
                        entity_type="core_value",
                        entity_id=cv.id,
                        entity_name=cv.name_ar,
                        field="definition",
                    )
                )

        # Check evidence if required
        if self.require_evidence and not cv.evidence:
            issues.append(
                ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    code="MISSING_EVIDENCE",
                    message=f"Core value '{cv.name_ar}' has no evidence references",
                    entity_type="core_value",
                    entity_id=cv.id,
                    entity_name=cv.name_ar,
                )
            )

        # Validate sub-values
        for sv in cv.sub_values:
            issues.extend(self._validate_sub_value(sv, cv))

        return issues

    def _validate_sub_value(
        self, sv: ExtractedSubValue, cv: ExtractedCoreValue
    ) -> list[ValidationIssue]:
        """Validate a sub-value."""
        issues = []

        # Check sub-value name
        if not sv.name_ar or not sv.name_ar.strip():
            issues.append(
                ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    code="MISSING_NAME",
                    message=f"Sub-value is missing Arabic name (under '{cv.name_ar}')",
                    entity_type="sub_value",
                    entity_id=sv.id,
                    field="name_ar",
                    para_index=sv.para_index,
                )
            )

        # Check definition if required (less strict for sub-values)
        if self.require_definitions:
            if not sv.definition or not sv.definition.text_ar.strip():
                issues.append(
                    ValidationIssue(
                        severity=ValidationSeverity.WARNING,
                        code="MISSING_DEFINITION",
                        message=f"Sub-value '{sv.name_ar}' is missing definition",
                        entity_type="sub_value",
                        entity_id=sv.id,
                        entity_name=sv.name_ar,
                        field="definition",
                    )
                )

        return issues

    def _check_duplicates(
        self, result: ExtractionResult
    ) -> list[ValidationIssue]:
        """Check for duplicate names under same parent."""
        issues = []

        if self.allow_duplicate_names:
            return issues

        # Check pillar name duplicates
        pillar_names = [p.name_ar.strip().lower() for p in result.pillars if p.name_ar]
        for name in set(pillar_names):
            if pillar_names.count(name) > 1:
                issues.append(
                    ValidationIssue(
                        severity=ValidationSeverity.ERROR,
                        code="DUPLICATE_NAME",
                        message=f"Duplicate pillar name: '{name}'",
                        entity_type="pillar",
                    )
                )

        # Check core value duplicates within each pillar
        for pillar in result.pillars:
            cv_names = [
                cv.name_ar.strip().lower()
                for cv in pillar.core_values
                if cv.name_ar
            ]
            for name in set(cv_names):
                if cv_names.count(name) > 1:
                    issues.append(
                        ValidationIssue(
                            severity=ValidationSeverity.ERROR,
                            code="DUPLICATE_NAME",
                            message=f"Duplicate core value name '{name}' in pillar '{pillar.name_ar}'",
                            entity_type="core_value",
                        )
                    )

            # Check sub-value duplicates within each core value
            for cv in pillar.core_values:
                sv_names = [
                    sv.name_ar.strip().lower()
                    for sv in cv.sub_values
                    if sv.name_ar
                ]
                for name in set(sv_names):
                    if sv_names.count(name) > 1:
                        issues.append(
                            ValidationIssue(
                                severity=ValidationSeverity.ERROR,
                                code="DUPLICATE_NAME",
                                message=f"Duplicate sub-value name '{name}' in '{cv.name_ar}'",
                                entity_type="sub_value",
                            )
                        )

        return issues


def validate_extraction(
    result: ExtractionResult,
    strict: bool = True,
) -> ValidationResult:
    """
    Convenience function to validate an extraction result.

    Args:
        result: The extraction result to validate.
        strict: Whether to use strict validation (require definitions).

    Returns:
        ValidationResult: Validation outcome.
    """
    validator = Validator(
        require_definitions=strict,
        require_evidence=False,  # Don't require evidence in MVP
    )
    return validator.validate(result)


def validate_evidence_refs(
    refs: list,
) -> list[ValidationIssue]:
    """
    Validate evidence references from parsing.

    Args:
        refs: List of parsed Quran or Hadith references.

    Returns:
        List of validation issues.
    """
    issues = []

    for ref in refs:
        if hasattr(ref, "parse_status"):
            if ref.parse_status == ParseStatus.FAILED:
                issues.append(
                    ValidationIssue(
                        severity=ValidationSeverity.ERROR,
                        code="PARSE_FAILED",
                        message=f"Failed to parse reference: {ref.ref_raw}",
                        field="evidence",
                    )
                )
            elif ref.parse_status == ParseStatus.NEEDS_REVIEW:
                issues.append(
                    ValidationIssue(
                        severity=ValidationSeverity.WARNING,
                        code="NEEDS_REVIEW",
                        message=f"Reference needs review: {ref.ref_raw}",
                        field="evidence",
                    )
                )

    return issues

