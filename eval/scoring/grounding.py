"""Grounding / hallucination scoring.

Operational definition:
- A claim is unsupported if it requires evidence and is not supported by at least
  one valid cited evidence span.

Hard fail:
- invalid citation spans (nonexistent chunk, out-of-bounds offsets, quote mismatch)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Optional

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from apps.api.retrieve.normalize_ar import extract_arabic_words, normalize_for_matching

from eval.types import ClaimSupportPolicy, EvalOutputRow


@dataclass(frozen=True)
class CitationValidity:
    valid: bool
    error: Optional[str] = None


async def _fetch_chunk_text(session: AsyncSession, chunk_id: str) -> Optional[str]:
    row = (
        await session.execute(
            text("SELECT text_ar FROM chunk WHERE chunk_id=:cid"),
            {"cid": chunk_id},
        )
    ).fetchone()
    if not row:
        return None
    return str(row.text_ar or "")


def _quote_words_ok(quote: str) -> bool:
    words = [w for w in (quote or "").split() if w]
    return len(words) <= 25


async def validate_citation(session: AsyncSession, citation: dict[str, Any]) -> CitationValidity:
    cid = str(citation.get("source_id") or "")
    if not cid:
        return CitationValidity(False, "missing source_id")

    txt = await _fetch_chunk_text(session, cid)
    if txt is None:
        return CitationValidity(False, f"chunk_id not found: {cid}")

    try:
        s = int(citation.get("span_start"))
        e = int(citation.get("span_end"))
    except Exception:
        return CitationValidity(False, f"invalid span offsets for {cid}")

    if s < 0 or e < 0 or e <= s:
        return CitationValidity(False, f"invalid span range {s}:{e} for {cid}")
    if e > len(txt):
        return CitationValidity(False, f"span out of bounds {s}:{e} for {cid} len={len(txt)}")

    quote = str(citation.get("quote") or "")
    if not _quote_words_ok(quote):
        return CitationValidity(False, f"quote too long (>25 words) for {cid}")

    # Quote must match substring content (normalized containment).
    sub = txt[s:e]
    nq = normalize_for_matching(quote)
    ns = normalize_for_matching(sub)
    if nq and nq not in ns:
        return CitationValidity(False, f"quote mismatch for {cid}")

    return CitationValidity(True, None)


def _term_coverage(claim_text: str, evidence_text: str, *, min_term_len: int = 3) -> float:
    terms = extract_arabic_words(claim_text or "")
    terms = [t for t in terms if len(t) >= min_term_len]
    if not terms:
        return 1.0
    ev_norm = normalize_for_matching(evidence_text or "")
    if not ev_norm:
        return 0.0

    covered = 0
    for t in terms:
        if normalize_for_matching(t) in ev_norm:
            covered += 1
    return covered / max(len(terms), 1)


async def claim_supported(session: AsyncSession, row: EvalOutputRow, claim: dict[str, Any]) -> bool:
    # Policy gating
    if not bool(claim.get("requires_evidence", True)):
        return True

    pol = claim.get("support_policy")
    if pol == ClaimSupportPolicy.NO_CITE_ALLOWED.value:
        return True
    if pol == ClaimSupportPolicy.MAY_CITE.value:
        # may_cite claims are allowed to be framework-bounded guidance.
        # They are tracked separately in reporting, not as grounding failures.
        return True

    spans = ((claim.get("evidence") or {}).get("supporting_spans") or [])
    if not spans:
        return False

    # Combine evidence texts if multi-span.
    combined = ""
    for sp in spans:
        v = await validate_citation(session, sp)
        if not v.valid:
            # Hard-fail handled elsewhere; for support check treat invalid as not supporting.
            continue
        cid = str(sp.get("source_id") or "")
        txt = await _fetch_chunk_text(session, cid)
        if txt is None:
            continue
        s = int(sp.get("span_start"))
        e = int(sp.get("span_end"))
        combined += " " + (txt[s:e] or "")

    if not combined.strip():
        return False

    cov = _term_coverage(str(claim.get("text_ar") or ""), combined)
    # Default strict threshold.
    return cov >= 0.5


@dataclass(frozen=True)
class GroundingMetrics:
    total_claims: int
    unsupported_claims: int
    unsupported_claim_rate: float

    citation_validity_errors: int

    # Abstention metrics (for mixed sets, not just negative-only)
    expected_abstain: int
    expected_answer: int
    abstained: int
    answered: int

    abstention_precision: float
    abstention_recall: float
    false_answer_rate: float
    false_abstention_rate: float


async def score_grounding(
    *,
    session: AsyncSession,
    outputs: list[EvalOutputRow],
    dataset_by_id: dict[str, dict[str, Any]],
    fail_on_invalid_citations: bool = True,
) -> GroundingMetrics:
    citation_errors = 0

    total_claims = 0
    unsupported_claims = 0

    expected_abstain = 0
    expected_answer = 0
    abstained = 0
    answered = 0

    tp_abstain = 0
    fp_abstain = 0
    fn_answer = 0

    for r in outputs:
        d = dataset_by_id.get(r.id, {})
        expect_abstain = bool(d.get("expect_abstain"))

        if expect_abstain:
            expected_abstain += 1
        else:
            expected_answer += 1

        if r.abstained:
            abstained += 1
            if expect_abstain:
                tp_abstain += 1
            else:
                fp_abstain += 1
        else:
            answered += 1
            if expect_abstain:
                fn_answer += 1

        # Validate citations
        for c in r.citations:
            v = await validate_citation(session, c.model_dump())
            if not v.valid:
                citation_errors += 1
                if fail_on_invalid_citations:
                    raise RuntimeError(f"Invalid citation for row={r.id}: {v.error}")

        # Claims
        for cl in r.claims:
            cld = cl.model_dump()
            if not bool(cld.get("requires_evidence", True)):
                continue
            if cld.get("support_policy") in {
                ClaimSupportPolicy.NO_CITE_ALLOWED.value,
                ClaimSupportPolicy.MAY_CITE.value,
            }:
                continue

            total_claims += 1
            ok = await claim_supported(session, r, cld)
            if not ok:
                unsupported_claims += 1

    rate = (unsupported_claims / total_claims) if total_claims else 0.0

    abstention_precision = (tp_abstain / (tp_abstain + fp_abstain)) if (tp_abstain + fp_abstain) else 1.0
    abstention_recall = (tp_abstain / (tp_abstain + fn_answer)) if (tp_abstain + fn_answer) else 1.0
    false_answer_rate = (fn_answer / expected_abstain) if expected_abstain else 0.0
    false_abstention_rate = (fp_abstain / expected_answer) if expected_answer else 0.0

    return GroundingMetrics(
        total_claims=total_claims,
        unsupported_claims=unsupported_claims,
        unsupported_claim_rate=rate,
        citation_validity_errors=citation_errors,
        expected_abstain=expected_abstain,
        expected_answer=expected_answer,
        abstained=abstained,
        answered=answered,
        abstention_precision=abstention_precision,
        abstention_recall=abstention_recall,
        false_answer_rate=false_answer_rate,
        false_abstention_rate=false_abstention_rate,
    )
