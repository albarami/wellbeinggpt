"""
Citation Enforcer Module

Implements strong verification guardrails:
1. CitationEnforcer - hard fail if citations=[] while not_found=false
2. Evidence-ID Verifier - fail if cited chunk_id not in evidence bundle
3. Claim-to-Evidence Checker - ensure key terms appear in cited evidence
"""

from dataclasses import dataclass
from typing import Any, Optional

from apps.api.retrieve.normalize_ar import extract_arabic_words, normalize_for_matching


REASONING_START = "[[MUHASIBI_REASONING_START]]"
REASONING_END = "[[MUHASIBI_REASONING_END]]"


def _strip_muhasibi_reasoning_block(text: str) -> str:
    """
    Remove Muḥāsibī reasoning metadata block from text.

    Reason: claim-to-evidence checking should evaluate only the *answer* content,
    not the user-visible methodology explanation block.
    """
    if not text:
        return ""

    out = text
    # Remove all occurrences deterministically (non-regex, bounded).
    for _ in range(3):  # hard cap to avoid pathological input
        start = out.find(REASONING_START)
        if start < 0:
            break
        end = out.find(REASONING_END, start)
        if end < 0:
            break
        end = end + len(REASONING_END)
        out = (out[:start] + out[end:]).strip()
    return out


@dataclass
class GuardrailResult:
    """Result of a guardrail check."""

    passed: bool
    issues: list[str]
    should_retry: bool = False
    should_refuse: bool = False


class CitationEnforcer:
    """
    Enforces citation requirements.

    Hard fail if:
    - Answer has 0 citations AND not_found=false
    """

    def check(
        self,
        answer_ar: str,
        citations: list[dict],
        not_found: bool,
    ) -> GuardrailResult:
        """
        Check if citation requirements are met.

        Args:
            answer_ar: The answer text.
            citations: List of citations.
            not_found: Whether the answer is a "not found" response.

        Returns:
            GuardrailResult indicating pass/fail.
        """
        issues = []

        # If not_found is true, no citations are required
        if not_found:
            return GuardrailResult(passed=True, issues=[])

        # Check for empty citations
        if not citations:
            issues.append("الإجابة لا تحتوي على استشهادات مع أن البيانات متوفرة")
            return GuardrailResult(
                passed=False,
                issues=issues,
                should_retry=True,
                should_refuse=False,
            )

        # Check for valid citation structure
        for i, citation in enumerate(citations):
            if not citation.get("chunk_id"):
                issues.append(f"الاستشهاد {i+1} لا يحتوي على معرف صالح")

        if issues:
            return GuardrailResult(
                passed=False,
                issues=issues,
                should_retry=True,
            )

        return GuardrailResult(passed=True, issues=[])


class EvidenceIdVerifier:
    """
    Verifies that cited chunk IDs exist in the evidence bundle.

    Fail if:
    - Any cited chunk_id is not present in the evidence packets
    """

    def check(
        self,
        citations: list[dict],
        evidence_packets: list[dict],
    ) -> GuardrailResult:
        """
        Verify that all cited IDs exist in evidence.

        Args:
            citations: List of citations from the answer.
            evidence_packets: List of evidence packets used.

        Returns:
            GuardrailResult indicating pass/fail.
        """
        issues = []

        # Build set of valid chunk IDs
        valid_ids = {p.get("chunk_id") for p in evidence_packets if p.get("chunk_id")}

        # Check each citation
        for citation in citations:
            chunk_id = citation.get("chunk_id", "")
            if chunk_id not in valid_ids:
                issues.append(f"الاستشهاد '{chunk_id}' غير موجود في الأدلة المتاحة")

        if issues:
            return GuardrailResult(
                passed=False,
                issues=issues,
                should_retry=True,
                should_refuse=False,
            )

        return GuardrailResult(passed=True, issues=[])


class ClaimToEvidenceChecker:
    """
    Checks that key terms in the answer appear in cited evidence.

    This is a deterministic check that:
    1. Extracts key terms from answer
    2. Verifies each term appears in at least one cited evidence chunk
    """

    def __init__(
        self,
        min_coverage_ratio: float = 0.5,
        min_term_length: int = 3,
    ):
        """
        Initialize the checker.

        Args:
            min_coverage_ratio: Minimum ratio of terms that must be covered.
            min_term_length: Minimum length of terms to check.
        """
        self.min_coverage_ratio = min_coverage_ratio
        self.min_term_length = min_term_length

    def check(
        self,
        answer_ar: str,
        citations: list[dict],
        evidence_packets: list[dict],
    ) -> GuardrailResult:
        """
        Check that answer terms are supported by evidence.

        Args:
            answer_ar: The answer text.
            citations: List of citations.
            evidence_packets: List of evidence packets.

        Returns:
            GuardrailResult indicating pass/fail.
        """
        issues = []

        # Ignore the Muḥāsibī reasoning block if present.
        answer_for_check = _strip_muhasibi_reasoning_block(answer_ar or "")

        # Extract key terms from answer
        answer_terms = extract_arabic_words(answer_for_check)
        answer_terms = [t for t in answer_terms if len(t) >= self.min_term_length]

        if not answer_terms:
            # No significant terms to check
            return GuardrailResult(passed=True, issues=[])

        # Build set of cited chunk IDs
        cited_ids = {c.get("chunk_id") for c in citations}

        # Build combined text from cited evidence
        cited_text = ""
        for packet in evidence_packets:
            if packet.get("chunk_id") in cited_ids:
                cited_text += " " + packet.get("text_ar", "")

        # Normalize cited text
        cited_normalized = normalize_for_matching(cited_text)

        # Check term coverage
        covered_terms = []
        uncovered_terms = []

        for term in answer_terms:
            normalized_term = normalize_for_matching(term)
            if normalized_term in cited_normalized:
                covered_terms.append(term)
            else:
                uncovered_terms.append(term)

        # Calculate coverage ratio
        if answer_terms:
            coverage_ratio = len(covered_terms) / len(answer_terms)
        else:
            coverage_ratio = 1.0

        # Check if coverage is sufficient
        if coverage_ratio < self.min_coverage_ratio:
            issues.append(
                f"تغطية المصطلحات غير كافية ({coverage_ratio:.0%}). "
                f"مصطلحات غير مدعومة: {', '.join(uncovered_terms[:5])}"
            )
            return GuardrailResult(
                passed=False,
                issues=issues,
                should_retry=True,
                should_refuse=len(covered_terms) == 0,
            )

        return GuardrailResult(passed=True, issues=[])


class Guardrails:
    """
    Combined guardrails for answer validation.

    Implements the two-pass output validation:
    - Pass A: GPT-5 returns JSON
    - Pass B: Deterministic verifier validates
    """

    def __init__(
        self,
        citation_enforcer: Optional[CitationEnforcer] = None,
        evidence_verifier: Optional[EvidenceIdVerifier] = None,
        claim_checker: Optional[ClaimToEvidenceChecker] = None,
    ):
        """
        Initialize guardrails.

        Args:
            citation_enforcer: Citation requirement checker.
            evidence_verifier: Evidence ID verifier.
            claim_checker: Claim-to-evidence checker.
        """
        self.citation_enforcer = citation_enforcer or CitationEnforcer()
        self.evidence_verifier = evidence_verifier or EvidenceIdVerifier()
        self.claim_checker = claim_checker or ClaimToEvidenceChecker()

    def validate(
        self,
        answer_ar: str,
        citations: list[dict],
        evidence_packets: list[dict],
        not_found: bool,
    ) -> GuardrailResult:
        """
        Run all guardrail checks.

        Args:
            answer_ar: The answer text.
            citations: List of citations.
            evidence_packets: List of evidence packets.
            not_found: Whether this is a "not found" response.

        Returns:
            Combined GuardrailResult.
        """
        all_issues = []

        # 1. Citation Enforcer
        citation_result = self.citation_enforcer.check(
            answer_ar, citations, not_found
        )
        if not citation_result.passed:
            all_issues.extend(citation_result.issues)
            if citation_result.should_refuse:
                return GuardrailResult(
                    passed=False,
                    issues=all_issues,
                    should_refuse=True,
                )

        # 2. Evidence ID Verifier
        if citations:
            evidence_result = self.evidence_verifier.check(
                citations, evidence_packets
            )
            if not evidence_result.passed:
                all_issues.extend(evidence_result.issues)
                if evidence_result.should_refuse:
                    return GuardrailResult(
                        passed=False,
                        issues=all_issues,
                        should_refuse=True,
                    )

        # 3. Claim-to-Evidence Checker
        if not not_found:
            claim_result = self.claim_checker.check(
                answer_ar, citations, evidence_packets
            )
            if not claim_result.passed:
                all_issues.extend(claim_result.issues)
                if claim_result.should_refuse:
                    return GuardrailResult(
                        passed=False,
                        issues=all_issues,
                        should_refuse=True,
                    )

        # If any issues, return them
        if all_issues:
            return GuardrailResult(
                passed=False,
                issues=all_issues,
                should_retry=True,
            )

        return GuardrailResult(passed=True, issues=[])

    def enforce_refusal(
        self,
        evidence_packets: list[dict],
    ) -> dict[str, Any]:
        """
        Generate a refusal response.

        Args:
            evidence_packets: Available evidence (shown even if insufficient).

        Returns:
            Refusal response data.
        """
        return {
            "answer_ar": "لا يوجد في البيانات الحالية ما يدعم الإجابة على هذا السؤال.",
            "not_found": True,
            "confidence": "low",
            "citations": [],
            "retrieved_packets": evidence_packets,
        }


def create_guardrails(
    min_coverage_ratio: float = 0.5,
) -> Guardrails:
    """
    Create a configured Guardrails instance.

    Args:
        min_coverage_ratio: Minimum term coverage required.

    Returns:
        Configured Guardrails instance.
    """
    return Guardrails(
        citation_enforcer=CitationEnforcer(),
        evidence_verifier=EvidenceIdVerifier(),
        claim_checker=ClaimToEvidenceChecker(
            min_coverage_ratio=min_coverage_ratio,
        ),
    )

