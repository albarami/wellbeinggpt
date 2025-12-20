"""Tests for integrity_validator.py - Sanity & Integrity Validator.

Reason: Ensure the system behaves like a scholar-critic that detects
malformed evidence and prevents propagation of mixed attributions.
"""

import pytest

from apps.api.core.integrity_validator import (
    IntegrityIssue,
    detect_mixed_attribution,
    validate_chunk,
    validate_evidence_packets,
    is_chunk_quarantined,
    get_quarantine_reason,
    get_integrity_warning_message,
)


class TestMixedAttributionDetection:
    """Tests for detecting mixed Quran/hadith attribution."""

    def test_detects_mixed_quran_hadith(self):
        """Should detect when Quran verse is mixed with hadith reference."""
        # The problematic chunk from framework
        text = "وأعدوا لهم ما استطعتم من قوة (صحيح مسلم 1917)"
        is_mixed, detail = detect_mixed_attribution(text)
        assert is_mixed is True
        assert detail is not None

    def test_pure_quran_not_flagged(self):
        """Pure Quran verse should not be flagged."""
        text = "وأعدوا لهم ما استطعتم من قوة [الأنفال:60]"
        is_mixed, _ = detect_mixed_attribution(text)
        assert is_mixed is False

    def test_pure_hadith_not_flagged(self):
        """Pure hadith should not be flagged."""
        text = "ألا إن القوة الرمي (صحيح مسلم 1917)"
        is_mixed, _ = detect_mixed_attribution(text)
        assert is_mixed is False

    def test_framework_prose_not_flagged(self):
        """Framework prose without verse patterns should not be flagged."""
        text = "الحياة الطيبة هي حياة متوازنة تراعي جميع أبعاد الإنسان"
        is_mixed, _ = detect_mixed_attribution(text)
        assert is_mixed is False


class TestChunkValidation:
    """Tests for validating individual chunks."""

    def test_quarantined_chunk_detected(self):
        """Known quarantined chunk should be flagged."""
        chunk = {
            "chunk_id": "CH_268ba301e082",
            "text_ar": "وأعدوا لهم ما استطعتم من قوة (صحيح مسلم 1917)",
        }
        result = validate_chunk(chunk)
        assert result.quarantined is True
        assert result.is_valid is False
        assert IntegrityIssue.MIXED_ATTRIBUTION in result.issues

    def test_valid_chunk_passes(self):
        """Valid chunk should pass validation."""
        chunk = {
            "chunk_id": "CH_valid_001",
            "text_ar": "الحياة الطيبة هي حياة متوازنة",
            "source_type": "framework_prose",
        }
        result = validate_chunk(chunk)
        assert result.is_valid is True
        assert result.quarantined is False
        assert len(result.issues) == 0


class TestEvidencePacketValidation:
    """Tests for validating lists of evidence packets."""

    def test_filters_quarantined_packets(self):
        """Should filter out quarantined packets from results."""
        packets = [
            {"chunk_id": "CH_268ba301e082", "text_ar": "وأعدوا..."},
            {"chunk_id": "CH_valid_001", "text_ar": "الحياة الطيبة..."},
        ]
        valid_packets, results = validate_evidence_packets(packets)
        
        # Should have filtered out the quarantined one
        assert len(valid_packets) == 1
        assert valid_packets[0]["chunk_id"] == "CH_valid_001"
        
        # But results should show both
        assert len(results) == 2

    def test_empty_packets_returns_empty(self):
        """Empty packet list should return empty results."""
        valid_packets, results = validate_evidence_packets([])
        assert len(valid_packets) == 0
        assert len(results) == 0


class TestQuarantineHelpers:
    """Tests for quarantine helper functions."""

    def test_is_chunk_quarantined(self):
        """Should correctly identify quarantined chunks."""
        assert is_chunk_quarantined("CH_268ba301e082") is True
        assert is_chunk_quarantined("CH_valid_001") is False

    def test_get_quarantine_reason(self):
        """Should return reason for quarantined chunks."""
        reason = get_quarantine_reason("CH_268ba301e082")
        assert reason is not None
        assert "Quran" in reason or "hadith" in reason.lower()
        
        reason = get_quarantine_reason("CH_valid_001")
        assert reason is None


class TestWarningMessages:
    """Tests for generating user-friendly warning messages."""

    def test_warning_for_quarantined(self):
        """Should generate Arabic warning for quarantined chunks."""
        from apps.api.core.integrity_validator import IntegrityResult
        
        results = [
            IntegrityResult(
                chunk_id="CH_268ba",
                is_valid=False,
                issues=[IntegrityIssue.MIXED_ATTRIBUTION],
                details=["Mixed attribution"],
                quarantined=True,
            )
        ]
        warning = get_integrity_warning_message(results)
        assert warning is not None
        assert "استبعاد" in warning  # Arabic for "excluded"

    def test_no_warning_for_valid(self):
        """Should return None when all chunks are valid."""
        from apps.api.core.integrity_validator import IntegrityResult
        
        results = [
            IntegrityResult(
                chunk_id="CH_valid",
                is_valid=True,
                issues=[],
                details=[],
                quarantined=False,
            )
        ]
        warning = get_integrity_warning_message(results)
        assert warning is None
