"""Deterministic post-generation pruning + fail-closed.

Policy:
- Any MUST_CITE claim with no supporting spans is removed.
- Any MUST_CITE claim that fails support check is removed.
- If pruning removes too much, abstain.

This module is used by the eval runner (not the production API).
"""

from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy.ext.asyncio import AsyncSession

from eval.scoring.grounding import claim_supported
from eval.types import ClaimSupportPolicy, EvalCitation, EvalClaim, EvalMode
from apps.api.retrieve.normalize_ar import extract_arabic_words, normalize_for_matching


@dataclass(frozen=True)
class PruneResult:
    answer_ar: str
    claims: list[EvalClaim]
    citations: list[EvalCitation]
    abstained: bool
    abstain_reason: str | None


def _join_sentences(sentences: list[str]) -> str:
    cleaned: list[str] = []
    for s in sentences:
        ss = (s or "").strip()
        if not ss:
            continue
        # Drop tiny fragments that often come from chunk formatting noise (e.g., a lone word heading).
        if len(ss) < 40 and len(extract_arabic_words(ss)) < 2:
            continue
        cleaned.append(ss)
    out = "\n".join(cleaned)
    return out.strip()


async def prune_and_fail_closed(
    *,
    session: AsyncSession,
    mode: EvalMode,
    answer_ar: str,
    claims: list[EvalClaim],
    citations: list[EvalCitation],
    question_ar: str,
    resolved_entities: list[dict[str, str]],
    required_graph_paths: list[dict] | None = None,
    min_sentences_remaining: int = 2,
) -> PruneResult:
    # Ungrounded mode isn't pruned.
    if mode == EvalMode.LLM_ONLY_UNGROUNDED:
        return PruneResult(
            answer_ar=answer_ar,
            claims=claims,
            citations=citations,
            abstained=False,
            abstain_reason=None,
        )

    kept_claims: list[EvalClaim] = []
    kept_sentences: list[str] = []

    # Deduce sentence order from claims (each claim is one sentence-like unit).
    for cl in claims:
        c = cl.model_dump()

        # Do not force evidence for may_cite or no_cite_allowed in pruning.
        pol = c.get("support_policy")
        if pol in {ClaimSupportPolicy.MAY_CITE.value, ClaimSupportPolicy.NO_CITE_ALLOWED.value}:
            kept_claims.append(cl)
            kept_sentences.append(cl.text_ar)
            continue

        # MUST_CITE: require at least one supporting span
        spans = ((c.get("evidence") or {}).get("supporting_spans") or [])
        if not spans:
            continue

        # claim_supported only uses the claim evidence spans + DB text, so row is unused.
        ok = await claim_supported(session, row=None, claim=c)  # type: ignore[arg-type]
        if not ok:
            continue

        kept_claims.append(cl)
        kept_sentences.append(cl.text_ar)

    pruned_answer = _join_sentences(kept_sentences)

    # Filter citations to only those referenced by kept_claims supporting spans.
    used_ids: set[tuple[str, int, int]] = set()
    for cl in kept_claims:
        for sp in cl.evidence.supporting_spans:
            used_ids.add((sp.source_id, sp.span_start, sp.span_end))
    kept_citations = [c for c in citations if (c.source_id, c.span_start, c.span_end) in used_ids]

    # If too little remains, abstain.
    if len(kept_sentences) < min_sentences_remaining:
        return PruneResult(
            answer_ar="لا يوجد في البيانات الحالية ما يدعم الإجابة على هذا السؤال.",
            claims=[],
            citations=[],
            abstained=True,
            abstain_reason="PRUNED_TOO_MUCH",
        )

    # Relationship completeness gate (post-gen, deterministic):
    # If the question asks for a relationship, but the answer doesn't mention at least
    # two distinct resolved entity names, abstain (prevents answering only one side).
    qn = normalize_for_matching(question_ar or "")
    looks_like_relationship = ("ما العلاقة" in qn) or ("العلاقة بين" in qn) or ("اربط بين" in qn) or ("ربط بين" in qn)
    # Dataset-required graph paths are an explicit signal the relationship is in-scope.
    # Do not fail-closed purely on name-mention heuristics in that case.
    if looks_like_relationship and resolved_entities and not (required_graph_paths or []):
        an = normalize_for_matching(pruned_answer)
        names = [str(e.get("name_ar") or "") for e in resolved_entities if (e.get("name_ar") or "").strip()]
        pair_keys = {
            (str(e.get("type") or ""), str(e.get("id") or ""))
            for e in resolved_entities
            if (e.get("type") and e.get("id"))
        }
        # Deduplicate by normalized form
        uniq = []
        seen = set()
        for n in names:
            nn = normalize_for_matching(n)
            if nn and nn not in seen:
                uniq.append(nn)
                seen.add(nn)
        mentioned = sum(1 for nn in uniq[:5] if nn in an)

        # If there are >=2 distinct entities but they share the same Arabic name (SAME_NAME),
        # require only mentioning the (shared) name at least once.
        required_mentions = 1 if (len(pair_keys) >= 2 and len(uniq) == 1) else 2
        if mentioned < required_mentions:
            return PruneResult(
                answer_ar="لا يوجد في البيانات الحالية ما يدعم الإجابة على هذا السؤال.",
                claims=[],
                citations=[],
                abstained=True,
                abstain_reason="RELATIONSHIP_INCOMPLETE",
            )

    return PruneResult(
        answer_ar=pruned_answer,
        claims=kept_claims,
        citations=kept_citations,
        abstained=False,
        abstain_reason=None,
    )
