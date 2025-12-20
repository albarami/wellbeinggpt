"""
Tests for framework structure extraction.

Verifies that the RuleExtractor correctly extracts the 5 pillars
and their expected core values and sub-values from the wellbeing framework DOCX.

These tests act as a regression gate to ensure extraction completeness.
"""

import pytest
from pathlib import Path

from apps.api.ingest.docx_reader import DocxReader
from apps.api.ingest.rule_extractor import RuleExtractor
from apps.api.ingest.canonical_json import extraction_to_canonical_json
from apps.api.ingest.pipeline_framework import _expand_evidence_in_canonical
from apps.api.retrieve.normalize_ar import normalize_for_matching


# Expected structure from the framework document
# Note: Actual names extracted may be simpler than full formal names
EXPECTED_PILLARS = [
    "الحياة الروحية",
    "الحياة العاطفية",
    "الحياة الفكرية",
    "الحياة البدنية",
    "الحياة الاجتماعية",
]

EXPECTED_SPIRITUAL_CORE_VALUES = [
    "الإيمان",
    "العبادة",  # May also be "العبادة والعبودية" in full form
    "التزكية",
]

EXPECTED_EMOTIONAL_CORE_VALUES = [
    "التوازن",
]

EXPECTED_INTELLECTUAL_CORE_VALUES_PARTIAL = [
    "التفكير",  # May include التفكير النقدي or variants
]

EXPECTED_PHYSICAL_CORE_VALUES_PARTIAL = [
    "الصحة",
    "القوة",
    "الوقاية",
]

EXPECTED_SOCIAL_CORE_VALUES_PARTIAL = [
    "الرعاية",
    "التعاون",
    "المسؤولية",
]

# Sample sub-values to check (not exhaustive)
EXPECTED_SPIRITUAL_SUB_VALUES_SAMPLE = [
    "التوحيد",
    "الإخلاص",
    "المراقبة",
]

EXPECTED_EMOTIONAL_SUB_VALUES_SAMPLE = [
    "إدراك المشاعر",
    "السكينة",
    "ضبط الانفعالات",
]


def normalize_name(name: str) -> str:
    """Normalize name for comparison."""
    return normalize_for_matching(name)


def find_pillar_by_name(pillars: list, name: str) -> dict | None:
    """Find a pillar by normalized name."""
    norm_target = normalize_name(name)
    for p in pillars:
        if norm_target in normalize_name(p.get("name_ar", "")):
            return p
    return None


def find_entity_by_name_partial(entities: list, partial_name: str) -> dict | None:
    """Find an entity where name contains the partial string."""
    norm_partial = normalize_name(partial_name)
    for e in entities:
        if norm_partial in normalize_name(e.get("name_ar", "")):
            return e
    return None


class TestFrameworkStructureExtraction:
    """Tests for framework structure extraction from DOCX."""

    @pytest.fixture(scope="class")
    def canonical_data(self):
        """Extract canonical data from the framework DOCX."""
        docx_path = Path("docs/source/framework_2025-10_v1.docx")
        if not docx_path.exists():
            pytest.skip("Framework DOCX not found")
        
        data = docx_path.read_bytes()
        parsed = DocxReader().read_bytes(data, docx_path.name)
        extracted = RuleExtractor(framework_version="2025-10").extract(parsed)
        canonical = _expand_evidence_in_canonical(extraction_to_canonical_json(extracted))
        return canonical

    def test_extracts_exactly_5_pillars(self, canonical_data):
        """Test that exactly 5 pillars are extracted."""
        pillars = canonical_data.get("pillars", [])
        assert len(pillars) == 5, f"Expected 5 pillars, got {len(pillars)}"

    def test_pillar_names_match_expected(self, canonical_data):
        """Test that pillar names match the expected Arabic names."""
        pillars = canonical_data.get("pillars", [])
        pillar_names = [p["name_ar"] for p in pillars]
        
        for expected in EXPECTED_PILLARS:
            found = any(
                normalize_name(expected) in normalize_name(name)
                for name in pillar_names
            )
            assert found, f"Expected pillar '{expected}' not found in {pillar_names}"

    def test_spiritual_pillar_core_values(self, canonical_data):
        """Test that spiritual pillar contains expected core values."""
        pillars = canonical_data.get("pillars", [])
        spiritual = find_pillar_by_name(pillars, "الروحية")
        
        assert spiritual is not None, "Spiritual pillar not found"
        
        core_values = spiritual.get("core_values", [])
        cv_names = [cv["name_ar"] for cv in core_values]
        
        for expected in EXPECTED_SPIRITUAL_CORE_VALUES:
            found = find_entity_by_name_partial(core_values, expected)
            assert found is not None, (
                f"Expected core value '{expected}' not found in spiritual pillar. "
                f"Found: {cv_names}"
            )

    def test_emotional_pillar_core_values(self, canonical_data):
        """Test that emotional pillar contains expected core values."""
        pillars = canonical_data.get("pillars", [])
        emotional = find_pillar_by_name(pillars, "العاطفية")
        
        assert emotional is not None, "Emotional pillar not found"
        
        core_values = emotional.get("core_values", [])
        cv_names = [cv["name_ar"] for cv in core_values]
        
        for expected in EXPECTED_EMOTIONAL_CORE_VALUES:
            found = find_entity_by_name_partial(core_values, expected)
            assert found is not None, (
                f"Expected core value '{expected}' not found in emotional pillar. "
                f"Found: {cv_names}"
            )

    def test_intellectual_pillar_has_core_values(self, canonical_data):
        """Test that intellectual pillar has core values."""
        pillars = canonical_data.get("pillars", [])
        intellectual = find_pillar_by_name(pillars, "الفكرية")
        
        assert intellectual is not None, "Intellectual pillar not found"
        
        core_values = intellectual.get("core_values", [])
        assert len(core_values) >= 1, "Intellectual pillar should have core values"

    def test_physical_pillar_core_values(self, canonical_data):
        """Test that physical pillar contains expected core values."""
        pillars = canonical_data.get("pillars", [])
        physical = find_pillar_by_name(pillars, "البدنية")
        
        assert physical is not None, "Physical pillar not found"
        
        core_values = physical.get("core_values", [])
        cv_names = [cv["name_ar"] for cv in core_values]
        
        # Check at least some expected core values are present
        found_count = sum(
            1 for expected in EXPECTED_PHYSICAL_CORE_VALUES_PARTIAL
            if find_entity_by_name_partial(core_values, expected)
        )
        assert found_count >= 1, (
            f"Expected at least one of {EXPECTED_PHYSICAL_CORE_VALUES_PARTIAL} "
            f"in physical pillar. Found: {cv_names}"
        )

    def test_social_pillar_core_values(self, canonical_data):
        """Test that social pillar contains expected core values."""
        pillars = canonical_data.get("pillars", [])
        social = find_pillar_by_name(pillars, "الاجتماعية")
        
        assert social is not None, "Social pillar not found"
        
        core_values = social.get("core_values", [])
        cv_names = [cv["name_ar"] for cv in core_values]
        
        # Check at least some expected core values are present
        found_count = sum(
            1 for expected in EXPECTED_SOCIAL_CORE_VALUES_PARTIAL
            if find_entity_by_name_partial(core_values, expected)
        )
        assert found_count >= 1, (
            f"Expected at least one of {EXPECTED_SOCIAL_CORE_VALUES_PARTIAL} "
            f"in social pillar. Found: {cv_names}"
        )

    def test_spiritual_pillar_has_sub_values(self, canonical_data):
        """Test that spiritual pillar core values have sub-values."""
        pillars = canonical_data.get("pillars", [])
        spiritual = find_pillar_by_name(pillars, "الروحية")
        
        assert spiritual is not None
        
        all_sub_values = []
        for cv in spiritual.get("core_values", []):
            all_sub_values.extend(cv.get("sub_values", []))
        
        sv_names = [sv["name_ar"] for sv in all_sub_values]
        
        for expected in EXPECTED_SPIRITUAL_SUB_VALUES_SAMPLE:
            found = find_entity_by_name_partial(all_sub_values, expected)
            assert found is not None, (
                f"Expected sub-value '{expected}' not found in spiritual pillar. "
                f"Found: {sv_names[:20]}..."
            )

    def test_emotional_pillar_has_sub_values(self, canonical_data):
        """Test that emotional pillar core values have sub-values."""
        pillars = canonical_data.get("pillars", [])
        emotional = find_pillar_by_name(pillars, "العاطفية")
        
        assert emotional is not None
        
        all_sub_values = []
        for cv in emotional.get("core_values", []):
            all_sub_values.extend(cv.get("sub_values", []))
        
        sv_names = [sv["name_ar"] for sv in all_sub_values]
        
        for expected in EXPECTED_EMOTIONAL_SUB_VALUES_SAMPLE:
            found = find_entity_by_name_partial(all_sub_values, expected)
            assert found is not None, (
                f"Expected sub-value '{expected}' not found in emotional pillar. "
                f"Found: {sv_names[:20]}..."
            )

    def test_total_core_values_reasonable(self, canonical_data):
        """Test that total core values count is reasonable."""
        total_cv = 0
        for p in canonical_data.get("pillars", []):
            total_cv += len(p.get("core_values", []))
        
        # Framework should have at least 10 core values
        assert total_cv >= 10, f"Expected at least 10 core values, got {total_cv}"

    def test_total_sub_values_reasonable(self, canonical_data):
        """Test that total sub-values count is reasonable."""
        total_sv = 0
        for p in canonical_data.get("pillars", []):
            for cv in p.get("core_values", []):
                total_sv += len(cv.get("sub_values", []))
        
        # Framework should have at least 30 sub-values
        assert total_sv >= 30, f"Expected at least 30 sub-values, got {total_sv}"

    def test_some_entities_have_definitions(self, canonical_data):
        """Test that some entities have definitions."""
        entities_with_def = 0
        
        for p in canonical_data.get("pillars", []):
            for cv in p.get("core_values", []):
                if cv.get("definition") and cv["definition"].get("text_ar"):
                    entities_with_def += 1
                for sv in cv.get("sub_values", []):
                    if sv.get("definition") and sv["definition"].get("text_ar"):
                        entities_with_def += 1
        
        # Should have many definitions
        assert entities_with_def >= 20, (
            f"Expected at least 20 entities with definitions, got {entities_with_def}"
        )

    def test_some_entities_have_evidence(self, canonical_data):
        """Test that some entities have evidence."""
        entities_with_evidence = 0
        
        for p in canonical_data.get("pillars", []):
            for cv in p.get("core_values", []):
                if cv.get("evidence"):
                    entities_with_evidence += 1
                for sv in cv.get("sub_values", []):
                    if sv.get("evidence"):
                        entities_with_evidence += 1
        
        # Should have some evidence
        assert entities_with_evidence >= 10, (
            f"Expected at least 10 entities with evidence, got {entities_with_evidence}"
        )


class TestCrossValueDetection:
    """Tests for cross-cutting value detection."""

    @pytest.fixture(scope="class")
    def canonical_data(self):
        """Extract canonical data from the framework DOCX."""
        docx_path = Path("docs/source/framework_2025-10_v1.docx")
        if not docx_path.exists():
            pytest.skip("Framework DOCX not found")
        
        data = docx_path.read_bytes()
        parsed = DocxReader().read_bytes(data, docx_path.name)
        extracted = RuleExtractor(framework_version="2025-10").extract(parsed)
        canonical = _expand_evidence_in_canonical(extraction_to_canonical_json(extracted))
        return canonical

    def test_sabr_exists_in_framework(self, canonical_data):
        """Test that الصبر (patience) exists as a sub-value."""
        all_sub_values = []
        for p in canonical_data.get("pillars", []):
            for cv in p.get("core_values", []):
                all_sub_values.extend(cv.get("sub_values", []))
        
        sabr = find_entity_by_name_partial(all_sub_values, "الصبر")
        assert sabr is not None, "الصبر (patience) should exist as a sub-value"




