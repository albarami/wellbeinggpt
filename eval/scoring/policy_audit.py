"""Claim-policy audit.

Final gate for "zero hallucination":
- Any declarative sentence in `answer_ar` must be represented by a MUST_CITE claim
  with >=1 supporting span.

Rationale:
- Prevents hiding hallucinations via misclassification (may_cite/no_cite_allowed).
- Ensures "unsupported_claim_rate (must_cite)=0" implies no unsupported answer sentences.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from apps.api.ingest.sentence_spans import sentence_spans, span_text
from apps.api.guardrails.citation_enforcer import REASONING_END, REASONING_START
from apps.api.retrieve.normalize_ar import normalize_for_matching

from eval.types import ClaimSupportPolicy, EvalOutputRow


@dataclass(frozen=True)
class PolicyAuditMetrics:
    audited_sentences: int
    violations: int
    violation_rate: float
    violations_by_reason: dict[str, int]


def _clean_answer(answer_ar: str) -> str:
    clean = (answer_ar or "").strip()
    for _ in range(3):
        s = clean.find(REASONING_START)
        if s < 0:
            break
        e = clean.find(REASONING_END, s)
        if e < 0:
            break
        e = e + len(REASONING_END)
        clean = (clean[:s] + clean[e:]).strip()
    return clean


def _is_exempt_sentence(sent: str) -> bool:
    """Return True for non-assertive/procedural/refusal sentences."""
    t = normalize_for_matching(sent or "")
    if not t:
        return True

    # Scholar-format section headers (structural, not factual claims)
    s = (sent or "").strip().strip(":：").strip()
    if s in {
        "مصفوفة المقارنة",
        "قسم (أ): ما يمكن دعمه من الأدلة المسترجعة",
        "قسم (ب): ما لا يمكن دعمه من الأدلة الحالية",
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
    }:
        return True

    # Pure refusal/meta
    if any(k in t for k in ["لا يوجد", "لا يمكن", "غير متوفر", "لا أستطيع", "لا اعلم"]):
        return True

    # Procedural/structural scaffolding
    if any(k in t for k in ["ساجيب", "سوف", "فيما يلي", "ملاحظه", "تنبيه", "وفق المصادر", "وفق النص"]):
        return True

    return False


def audit_policy_row(row: EvalOutputRow) -> tuple[PolicyAuditMetrics, list[dict[str, Any]]]:
    """Audit one output row; returns metrics + violation examples."""
    if row.abstained:
        return PolicyAuditMetrics(0, 0, 0.0, {}), []

    answer = _clean_answer(row.answer_ar)
    spans = sentence_spans(answer)

    by_norm_claim = {normalize_for_matching(c.text_ar): c for c in (row.claims or []) if (c.text_ar or "").strip()}

    audited = 0
    violations = 0
    reasons: dict[str, int] = {}
    examples: list[dict[str, Any]] = []

    for sp in spans:
        sent = (span_text(answer, sp) or "").strip()
        if not sent:
            continue
        if len([ch for ch in sent if ch.isalnum()]) < 3:
            continue
        if _is_exempt_sentence(sent):
            continue

        audited += 1
        key = normalize_for_matching(sent)
        cl = by_norm_claim.get(key)
        if cl is None:
            violations += 1
            reasons["MISSING_CLAIM"] = reasons.get("MISSING_CLAIM", 0) + 1
            if len(examples) < 20:
                examples.append({"id": row.id, "sentence": sent, "reason": "MISSING_CLAIM"})
            continue

        if cl.support_policy != ClaimSupportPolicy.MUST_CITE:
            violations += 1
            reasons["POLICY_NOT_MUST_CITE"] = reasons.get("POLICY_NOT_MUST_CITE", 0) + 1
            if len(examples) < 20:
                examples.append(
                    {
                        "id": row.id,
                        "sentence": sent,
                        "reason": "POLICY_NOT_MUST_CITE",
                        "support_policy": cl.support_policy.value,
                    }
                )
            continue

        spans_list = (cl.evidence.supporting_spans or [])
        if not spans_list:
            violations += 1
            reasons["NO_EVIDENCE_SPANS"] = reasons.get("NO_EVIDENCE_SPANS", 0) + 1
            if len(examples) < 20:
                examples.append({"id": row.id, "sentence": sent, "reason": "NO_EVIDENCE_SPANS"})
            continue

    rate = (violations / audited) if audited else 0.0
    return PolicyAuditMetrics(audited, violations, rate, reasons), examples


def score_policy_audit(outputs: list[EvalOutputRow]) -> tuple[PolicyAuditMetrics, list[dict[str, Any]]]:
    total_audited = 0
    total_violations = 0
    reasons: dict[str, int] = {}
    examples: list[dict[str, Any]] = []

    for r in outputs:
        m, ex = audit_policy_row(r)
        total_audited += m.audited_sentences
        total_violations += m.violations
        for k, v in (m.violations_by_reason or {}).items():
            reasons[k] = reasons.get(k, 0) + int(v)
        if len(examples) < 20:
            examples.extend(ex[: max(0, 20 - len(examples))])

    rate = (total_violations / total_audited) if total_audited else 0.0
    return PolicyAuditMetrics(total_audited, total_violations, rate, reasons), examples
