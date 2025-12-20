"""Deterministic claim extraction for eval outputs.

We intentionally keep this conservative:
- split answer into sentence-like units
- each sentence becomes one claim
- claim evidence mapping is explicit: by default, attach all available citations

Scorers then decide whether a claim is supported.
"""

from __future__ import annotations

import hashlib
from typing import Iterable

from apps.api.ingest.sentence_spans import sentence_spans, span_text
from apps.api.retrieve.normalize_ar import normalize_for_matching, extract_arabic_words
from apps.api.guardrails.citation_enforcer import REASONING_END, REASONING_START
from eval.types import (
    ClaimEvidenceBinding,
    ClaimSupportPolicy,
    ClaimSupportStrength,
    EvalCitation,
    EvalClaim,
    EvalMode,
)


def _stable_claim_id(text_ar: str) -> str:
    h = hashlib.sha256((text_ar or "").encode("utf-8")).hexdigest()[:12]
    return f"cl_{h}"


def extract_claims(
    *,
    answer_ar: str,
    mode: EvalMode,
    citations: list[EvalCitation],
) -> list[EvalClaim]:
    if not (answer_ar or "").strip():
        return []

    # Remove Muḥāsibī reasoning block before extracting claims.
    # Reason: the reasoning methodology block is not a factual answer and would
    # create many false \"unsupported\" claims.
    clean = answer_ar
    for _ in range(3):
        s = clean.find(REASONING_START)
        if s < 0:
            break
        e = clean.find(REASONING_END, s)
        if e < 0:
            break
        e = e + len(REASONING_END)
        clean = (clean[:s] + clean[e:]).strip()

    spans = sentence_spans(clean)
    claims: list[EvalClaim] = []

    def _is_section_header(sentence: str) -> bool:
        """
        Detect scholar-format section headers that should not require citations.

        Reason: headers are structural; forcing MUST_CITE would cause pruning to
        delete them and collapse rubric-visible structure.
        """
        s = (sentence or "").strip().strip(":：").strip()
        if not s:
            return False
        headers = {
            "مصفوفة المقارنة",
            "قسم (أ): ما يمكن دعمه من الأدلة المسترجعة",
            "قسم (ب): ما لا يمكن دعمه من الأدلة الحالية",
            # New naturalized partial headings (no A/B labels)
            "ما يمكن دعمه من الأدلة المسترجعة",
            "ما لا يمكن الجزم به من الأدلة الحالية",
            "تعريف المفهوم داخل الإطار",
            "التأصيل والأدلة",
            "التأصيل والأدلة (مختصر ومركز)",
            "الربط بين الركائز",
            "الربط بين الركائز (مع سبب الربط)",
            "تطبيق عملي",
            "تطبيق عملي على الحالة/السؤال",
            "تنبيهات وأخطاء شائعة",
            "خلاصة تنفيذية",
            "خلاصة تنفيذية (3 نقاط)",
            # Deterministic fallback headings (legacy)
            "التعريف (من النص)",
            "الدليل/التأصيل (من النص)",
            "تطبيق/إرشادات داخل الإطار (من النص)",
        }
        return s in headers

    def _bind_spans(sentence: str) -> list[EvalCitation]:
        """Bind a sentence to supporting spans by deterministic term overlap."""
        sent_norm = normalize_for_matching(sentence)
        if not sent_norm:
            return []
        sent_terms = [t for t in extract_arabic_words(sentence) if len(t) >= 3]
        if not sent_terms:
            return []
        out: list[EvalCitation] = []
        for c in citations:
            qn = normalize_for_matching(c.quote)
            if not qn:
                continue
            hit = 0
            for t in sent_terms[:20]:
                if normalize_for_matching(t) in qn:
                    hit += 1
            if hit >= 1:
                out.append(c)
        return out

    for sp in spans:
        sent = (span_text(clean, sp) or "").strip()
        if not sent:
            continue
        # Skip punctuation-only / trivial fragments.
        if len([ch for ch in sent if ch.isalnum()]) < 3:
            continue

        # Policy: ungrounded baseline is excluded from grounding KPIs.
        requires_evidence = mode != EvalMode.LLM_ONLY_UNGROUNDED

        # Conservative policy (to make "must_cite=0 unsupported" equivalent to "0 unsupported sentences"):
        # - For grounded modes: treat ALL non-meta sentences as MUST_CITE.
        # - Only allow NO_CITE_ALLOWED for pure refusal/meta statements and structure headers.
        # Meta/refusal/placeholder sentences that should not require citations.
        # Reason: these are either refusals or explicit "not in sources" flags.
        is_meta = any(
            k in sent
            for k in [
                "لا أعلم",
                "لا يوجد",
                "لا يمكن",
                "غير متوفر",
                "لا أستطيع",
                "لا توجد روابط",
                "غير منصوص عليه",
            ]
        )
        is_header = _is_section_header(sent)
        claim_type = "meta" if (is_meta or is_header) else "fact"

        if not requires_evidence:
            policy = ClaimSupportPolicy.MAY_CITE
            strength = ClaimSupportStrength.NONE_ALLOWED
        else:
            if claim_type == "meta":
                policy = ClaimSupportPolicy.NO_CITE_ALLOWED
                strength = ClaimSupportStrength.NONE_ALLOWED
            else:
                policy = ClaimSupportPolicy.MUST_CITE
                strength = ClaimSupportStrength.MULTI_SPAN if len(citations) > 1 else ClaimSupportStrength.DIRECT

        bound = _bind_spans(sent) if policy == ClaimSupportPolicy.MUST_CITE else []
        claims.append(
            EvalClaim(
                claim_id=_stable_claim_id(sent),
                text_ar=sent,
                support_strength=strength,
                support_policy=policy,
                evidence=ClaimEvidenceBinding(supporting_spans=list(bound)),
                requires_evidence=requires_evidence,
                claim_type=claim_type,  # type: ignore[arg-type]
            )
        )

    return claims
