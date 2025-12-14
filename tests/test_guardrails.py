"""
Tests for guardrails module.

Tests CitationEnforcer, EvidenceIdVerifier, and ClaimToEvidenceChecker.
"""

import pytest

from apps.api.guardrails.citation_enforcer import (
    CitationEnforcer,
    EvidenceIdVerifier,
    ClaimToEvidenceChecker,
    Guardrails,
    GuardrailResult,
    create_guardrails,
)


class TestCitationEnforcer:
    """Tests for CitationEnforcer."""

    def test_passes_when_not_found(self):
        """Test that not_found=true bypasses citation check."""
        enforcer = CitationEnforcer()
        result = enforcer.check(
            answer_ar="لا توجد إجابة",
            citations=[],
            not_found=True,
        )

        assert result.passed is True
        assert len(result.issues) == 0

    def test_fails_when_no_citations_with_answer(self):
        """Test that empty citations with not_found=false fails."""
        enforcer = CitationEnforcer()
        result = enforcer.check(
            answer_ar="هذه إجابة بدون استشهادات",
            citations=[],
            not_found=False,
        )

        assert result.passed is False
        assert len(result.issues) >= 1
        assert result.should_retry is True

    def test_passes_with_valid_citations(self):
        """Test that valid citations pass."""
        enforcer = CitationEnforcer()
        result = enforcer.check(
            answer_ar="الإيمان هو التصديق",
            citations=[{"chunk_id": "CH_000001", "source_anchor": "p1"}],
            not_found=False,
        )

        assert result.passed is True

    def test_fails_with_invalid_citation_structure(self):
        """Test that citations without chunk_id fail."""
        enforcer = CitationEnforcer()
        result = enforcer.check(
            answer_ar="إجابة",
            citations=[{"source_anchor": "p1"}],  # Missing chunk_id
            not_found=False,
        )

        assert result.passed is False


class TestEvidenceIdVerifier:
    """Tests for EvidenceIdVerifier."""

    def test_passes_when_all_ids_exist(self):
        """Test that valid chunk_ids pass."""
        verifier = EvidenceIdVerifier()

        evidence = [
            {"chunk_id": "CH_000001", "text_ar": "نص"},
            {"chunk_id": "CH_000002", "text_ar": "نص آخر"},
        ]

        citations = [
            {"chunk_id": "CH_000001"},
            {"chunk_id": "CH_000002"},
        ]

        result = verifier.check(citations, evidence)

        assert result.passed is True

    def test_fails_when_id_missing(self):
        """Test that missing chunk_id fails."""
        verifier = EvidenceIdVerifier()

        evidence = [
            {"chunk_id": "CH_000001", "text_ar": "نص"},
        ]

        citations = [
            {"chunk_id": "CH_000001"},
            {"chunk_id": "CH_999999"},  # Not in evidence
        ]

        result = verifier.check(citations, evidence)

        assert result.passed is False
        assert "CH_999999" in result.issues[0]

    def test_passes_with_empty_citations(self):
        """Test that empty citations pass (no IDs to verify)."""
        verifier = EvidenceIdVerifier()
        result = verifier.check([], [{"chunk_id": "CH_000001"}])

        assert result.passed is True


class TestClaimToEvidenceChecker:
    """Tests for ClaimToEvidenceChecker."""

    def test_passes_when_terms_covered(self):
        """Test that answer terms in evidence pass."""
        checker = ClaimToEvidenceChecker(min_coverage_ratio=0.5)

        answer = "الإيمان هو التصديق بالقلب"
        citations = [{"chunk_id": "CH_000001"}]
        evidence = [
            {
                "chunk_id": "CH_000001",
                "text_ar": "الإيمان هو التصديق بالقلب والإقرار باللسان",
            }
        ]

        result = checker.check(answer, citations, evidence)

        assert result.passed is True

    def test_fails_when_terms_not_covered(self):
        """Test that uncovered terms cause failure."""
        checker = ClaimToEvidenceChecker(min_coverage_ratio=0.8)

        answer = "الصلاة والزكاة والحج والصوم من أركان الإسلام"
        citations = [{"chunk_id": "CH_000001"}]
        evidence = [
            {
                "chunk_id": "CH_000001",
                "text_ar": "الصلاة ركن من أركان الإسلام",  # Only mentions الصلاة
            }
        ]

        result = checker.check(answer, citations, evidence)

        assert result.passed is False
        assert result.should_retry is True

    def test_passes_with_empty_answer(self):
        """Test that empty answer passes (nothing to check)."""
        checker = ClaimToEvidenceChecker()
        result = checker.check("", [], [])

        assert result.passed is True

    def test_only_checks_cited_evidence(self):
        """Test that only cited evidence is checked."""
        checker = ClaimToEvidenceChecker(min_coverage_ratio=0.5)

        answer = "الإيمان"
        citations = [{"chunk_id": "CH_000001"}]
        evidence = [
            {"chunk_id": "CH_000001", "text_ar": "الإيمان"},
            {"chunk_id": "CH_000002", "text_ar": "نص آخر مختلف تماما"},
        ]

        result = checker.check(answer, citations, evidence)

        assert result.passed is True

    def test_ignores_muhasibi_reasoning_block_markers(self):
        """Terms inside the reasoning block should not be claim-checked."""
        checker = ClaimToEvidenceChecker(min_coverage_ratio=0.8)

        answer = (
            "[[MUHASIBI_REASONING_START]]\n"
            "تفكير المحاسبي (عرض منهجي):\n"
            "- مصطلح_غير_مدعوم\n"
            "[[MUHASIBI_REASONING_END]]\n"
            "التقبل هو الاعتراف بالواقع كما هو."
        )
        citations = [{"chunk_id": "CH_000001"}]
        evidence = [{"chunk_id": "CH_000001", "text_ar": "التقبل هو الاعتراف بالواقع كما هو"}]

        result = checker.check(answer, citations, evidence)
        assert result.passed is True

    def test_still_fails_when_uncovered_terms_outside_reasoning_block(self):
        """Uncovered terms outside the reasoning block must still fail."""
        checker = ClaimToEvidenceChecker(min_coverage_ratio=0.8)

        answer = (
            "[[MUHASIBI_REASONING_START]]\n"
            "تفكير المحاسبي (عرض منهجي):\n"
            "- ملاحظة\n"
            "[[MUHASIBI_REASONING_END]]\n"
            "مصطلح_غير_مدعوم خارج الكتلة."
        )
        citations = [{"chunk_id": "CH_000001"}]
        evidence = [{"chunk_id": "CH_000001", "text_ar": "نص لا يحتوي على المصطلح"}]

        result = checker.check(answer, citations, evidence)
        assert result.passed is False
        assert result.should_retry is True


class TestGuardrails:
    """Tests for combined Guardrails."""

    def test_all_checks_pass(self):
        """Test that all checks passing results in overall pass."""
        guardrails = create_guardrails()

        result = guardrails.validate(
            answer_ar="الإيمان هو التصديق",
            citations=[{"chunk_id": "CH_000001"}],
            evidence_packets=[{"chunk_id": "CH_000001", "text_ar": "الإيمان هو التصديق بالقلب"}],
            not_found=False,
        )

        assert result.passed is True

    def test_refusal_when_no_citations(self):
        """Test that missing citations trigger retry or refuse."""
        guardrails = create_guardrails()

        result = guardrails.validate(
            answer_ar="إجابة بدون استشهاد",
            citations=[],
            evidence_packets=[{"chunk_id": "CH_000001", "text_ar": "نص"}],
            not_found=False,
        )

        assert result.passed is False
        # May trigger retry or refuse depending on coverage
        assert result.should_retry is True or result.should_refuse is True

    def test_not_found_bypasses_checks(self):
        """Test that not_found=true bypasses most checks."""
        guardrails = create_guardrails()

        result = guardrails.validate(
            answer_ar="لا توجد بيانات",
            citations=[],
            evidence_packets=[],
            not_found=True,
        )

        assert result.passed is True

    def test_enforce_refusal(self):
        """Test the refusal response generation."""
        guardrails = create_guardrails()

        evidence = [{"chunk_id": "CH_000001", "text_ar": "بعض النص"}]
        refusal = guardrails.enforce_refusal(evidence)

        assert refusal["not_found"] is True
        assert refusal["citations"] == []
        assert len(refusal["answer_ar"]) > 0
        assert refusal["retrieved_packets"] == evidence


class TestGuardrailsIntegration:
    """Integration tests for guardrails in realistic scenarios."""

    def test_in_corpus_answer_with_citations(self):
        """Test a valid in-corpus answer with proper citations."""
        guardrails = create_guardrails(min_coverage_ratio=0.3)

        answer = "الإيمان هو التصديق بالقلب والإقرار باللسان"
        citations = [{"chunk_id": "CH_000001"}]
        evidence = [
            {
                "chunk_id": "CH_000001",
                "text_ar": "الإيمان هو التصديق بالقلب والإقرار باللسان والعمل بالأركان",
            }
        ]

        result = guardrails.validate(answer, citations, evidence, not_found=False)

        assert result.passed is True
        assert len(result.issues) == 0

    def test_out_of_corpus_refusal(self):
        """Test proper handling of out-of-corpus questions."""
        guardrails = create_guardrails()

        # Simulate empty evidence (out of corpus)
        result = guardrails.validate(
            answer_ar="لا يوجد في البيانات الحالية ما يدعم الإجابة",
            citations=[],
            evidence_packets=[],
            not_found=True,
        )

        assert result.passed is True

    def test_hallucination_attempt_blocked(self):
        """Test that answer with uncited claims fails."""
        guardrails = create_guardrails(min_coverage_ratio=0.5)

        # Answer contains terms not in evidence
        answer = "الصبر والشكر من أعظم القيم في الإسلام ويرتبطان بالإيمان"
        citations = [{"chunk_id": "CH_000001"}]
        evidence = [
            {
                "chunk_id": "CH_000001",
                "text_ar": "الإيمان قيمة عظيمة",  # Doesn't mention الصبر or الشكر
            }
        ]

        result = guardrails.validate(answer, citations, evidence, not_found=False)

        # Should fail due to uncovered terms
        assert result.passed is False

