from __future__ import annotations

from eval.scoring.policy_audit import score_policy_audit
from eval.types import (
    ClaimEvidenceBinding,
    ClaimSupportPolicy,
    ClaimSupportStrength,
    EvalCitation,
    EvalClaim,
    EvalMode,
    EvalOutputRow,
    GraphTrace,
    RetrievalTrace,
)


def test_policy_audit_flags_non_must_cite_sentence() -> None:
    row = EvalOutputRow(
        id="x",
        mode=EvalMode.RAG_ONLY,
        question="س",
        answer_ar="هذه جملة تقريرية.",
        claims=[
            EvalClaim(
                claim_id="c1",
                text_ar="هذه جملة تقريرية.",
                support_strength=ClaimSupportStrength.DIRECT,
                support_policy=ClaimSupportPolicy.NO_CITE_ALLOWED,
                evidence=ClaimEvidenceBinding(supporting_spans=[]),
                requires_evidence=True,
                claim_type="fact",
            )
        ],
        citations=[EvalCitation(source_id="CH_x", span_start=0, span_end=1, quote="q")],
        retrieval_trace=RetrievalTrace(),
        graph_trace=GraphTrace(),
        almuhasbi_trace=None,
        abstained=False,
        abstain_reason=None,
        latency_ms=1,
        debug={},
    )

    m, examples = score_policy_audit([row])
    assert m.violations == 1
    assert examples


def test_policy_audit_passes_when_must_cite_with_spans() -> None:
    row = EvalOutputRow(
        id="x",
        mode=EvalMode.RAG_ONLY,
        question="س",
        answer_ar="هذه جملة تقريرية.",
        claims=[
            EvalClaim(
                claim_id="c1",
                text_ar="هذه جملة تقريرية.",
                support_strength=ClaimSupportStrength.DIRECT,
                support_policy=ClaimSupportPolicy.MUST_CITE,
                evidence=ClaimEvidenceBinding(
                    supporting_spans=[EvalCitation(source_id="CH_x", span_start=0, span_end=1, quote="q")]
                ),
                requires_evidence=True,
                claim_type="fact",
            )
        ],
        citations=[EvalCitation(source_id="CH_x", span_start=0, span_end=1, quote="q")],
        retrieval_trace=RetrievalTrace(),
        graph_trace=GraphTrace(),
        almuhasbi_trace=None,
        abstained=False,
        abstain_reason=None,
        latency_ms=1,
        debug={},
    )

    m, _ = score_policy_audit([row])
    assert m.violations == 0
