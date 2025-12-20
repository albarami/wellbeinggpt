"""Deterministic answer-eligibility (OOS) policy.

Goal: decide whether the system should answer or abstain *before generation*.

This is used by the eval runner for all grounded modes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from apps.api.core.muhasibi_listen import _detect_out_of_scope_ar
from apps.api.retrieve.normalize_ar import extract_arabic_words, normalize_for_matching

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession


@dataclass(frozen=True)
class EligibilityDecision:
    should_answer: bool
    reason_code: str


@dataclass(frozen=True)
class EligibilityThresholds:
    # Retrieval score thresholds (empirical; MergeRanker scores are ~0.5-1.2)
    top1_min: float = 0.75
    mean_topk_min: float = 0.65

    # Entity confidence threshold to consider in-scope anchoring
    entity_conf_min: float = 0.80


def _mean(xs: list[float]) -> float:
    if not xs:
        return 0.0
    return sum(xs) / len(xs)


def decide_answer_eligibility(
    *,
    question_ar: str,
    resolved_entities: list[dict[str, Any]],
    retrieval_trace: dict[str, Any],
    mode: str,
    thresholds: EligibilityThresholds = EligibilityThresholds(),
) -> EligibilityDecision:
    """Return whether to answer or abstain.

    Deterministic features only:
    - OOS keyword heuristics (framework-scoped)
    - entity resolution confidence
    - retrieval score signals
    """
    qn = normalize_for_matching(question_ar or "")

    # 0) Explicit out-of-scope markers
    oos, _reason = _detect_out_of_scope_ar(qn)
    if oos:
        return EligibilityDecision(False, "OOS_KEYWORDS")

    # 0b) Definition/biography requests with no resolved entities -> abstain.
    # Reason: prevents answering plausible-sounding but nonexistent framework terms.
    has_entities = bool(resolved_entities)
    looks_like_definition = ("عرّف" in qn) or ("تعريف" in qn) or ("كما ورد" in qn and "الاطار" in qn)
    looks_like_biography = qn.startswith("من هو") or ("من هو" in qn)
    looks_like_relationship = ("ما العلاقة" in qn) or ("العلاقة بين" in qn) or ("اربط بين" in qn) or ("ربط بين" in qn)
    if (looks_like_definition or looks_like_biography or looks_like_relationship) and not has_entities:
        return EligibilityDecision(False, "NO_ENTITY_MATCH")

    # Relationship questions should resolve at least two entities; otherwise we might be matching only one endpoint
    # (e.g., a real entity + a fake term) and hallucinating the other side.
    if looks_like_relationship and len(resolved_entities) < 2:
        return EligibilityDecision(False, "INSUFFICIENT_ENTITIES_FOR_RELATIONSHIP")

    # Relationship questions are high-risk for partial/overmatching.
    # If any entity match is low confidence or fuzzy, abstain to avoid fabricating links.
    if looks_like_relationship:
        # Require that at least two resolved entity display names actually appear in the question.
        # Reason: prevents mapping fake terms to a different real entity via fuzzy matching.
        qn_norm = qn
        name_hits = 0
        for e in resolved_entities[:5]:
            name = str(e.get("name_ar") or "").strip()
            if name and normalize_for_matching(name) in qn_norm:
                name_hits += 1
        if name_hits < 2:
            return EligibilityDecision(False, "RELATIONSHIP_NAMES_NOT_IN_QUESTION")

        # Additionally, ensure the two relationship endpoints in the text can be mapped
        # to resolved entities with high phrase-overlap (not just a shared keyword).
        # This protects against questions like "الاتزان الكوني" matching an entity "الاتزان".
        try:
            after = qn_norm.split("بين", 1)[1]
            if "وبين" in after:
                left_raw, _, right_raw = after.partition("وبين")
            else:
                # Fallback: split on " و " once.
                left_raw, _, right_raw = after.partition(" و ")
            # Stop at common trailing request phrase.
            for cut in ["اعطني", "أعطني", "شواهد", "من النص"]:
                if cut in right_raw:
                    right_raw = right_raw.split(cut, 1)[0]
                if cut in left_raw:
                    left_raw = left_raw.split(cut, 1)[0]
            left = left_raw.strip(" ؟?،. ").strip()
            right = right_raw.strip(" ؟?،. ").strip()
            prefixes = {"قيمة", "مبدأ", "ركيزة", "مفهوم", "القيمة", "المبدأ", "الركيزة"}
            left_terms = [t for t in extract_arabic_words(left) if t and normalize_for_matching(t) not in prefixes]
            right_terms = [t for t in extract_arabic_words(right) if t and normalize_for_matching(t) not in prefixes]
            left_set = set([normalize_for_matching(t) for t in left_terms if len(t) >= 2])
            right_set = set([normalize_for_matching(t) for t in right_terms if len(t) >= 2])
            if left_set and right_set:
                best_left = 0.0
                best_right = 0.0
                for e in resolved_entities[:5]:
                    en = normalize_for_matching(str(e.get("name_ar") or ""))
                    e_set = set([t for t in en.split() if t])
                    if not e_set:
                        continue
                    inter_left = len(left_set.intersection(e_set))
                    inter_right = len(right_set.intersection(e_set))
                    best_left = max(best_left, inter_left / max(len(left_set), len(e_set), 1))
                    best_right = max(best_right, inter_right / max(len(right_set), len(e_set), 1))
                if best_left < 0.8 or best_right < 0.8:
                    return EligibilityDecision(False, "RELATIONSHIP_ENDPOINT_MISMATCH")
        except Exception:
            # Best-effort; if parsing fails we fall back to other gates.
            pass

        for e in resolved_entities[:5]:
            try:
                conf = float(e.get("confidence") or 0.0)
            except Exception:
                conf = 0.0
            mt = str(e.get("match_type") or "")
            # Relationship questions are especially sensitive to over-matching.
            # Require *exact* endpoints only (or alias_exact).
            if conf < 0.95:
                return EligibilityDecision(False, "LOW_CONF_ENTITY_IN_RELATIONSHIP")
            if not (mt == "exact" or mt == "alias_exact"):
                return EligibilityDecision(False, "NON_EXACT_ENTITY_IN_RELATIONSHIP")

    # 1) Retrieval score signals
    topk = retrieval_trace.get("top_k_chunks") or []
    scores: list[float] = []
    for t in topk:
        try:
            scores.append(float(t.get("score") or 0.0))
        except Exception:
            continue

    top1 = scores[0] if scores else 0.0
    mean_top = _mean(scores[: min(10, len(scores))])

    # 2) Entity anchoring
    best_conf = 0.0
    try:
        for e in resolved_entities[:5]:
            best_conf = max(best_conf, float(e.get("confidence") or 0.0))
    except Exception:
        best_conf = 0.0

    # 3) Decide
    # If retrieval is weak AND entity anchoring is weak => abstain.
    if top1 < thresholds.top1_min and mean_top < thresholds.mean_topk_min and best_conf < thresholds.entity_conf_min:
        return EligibilityDecision(False, "LOW_RETRIEVAL_LOW_ENTITY")

    # If no retrieved chunks at all, abstain.
    if not topk:
        return EligibilityDecision(False, "NO_RETRIEVAL")

    return EligibilityDecision(True, "OK")


async def decide_answer_eligibility_with_db(
    *,
    session: AsyncSession,
    question_ar: str,
    retrieval_trace: dict[str, Any],
    resolved_entities: list[dict[str, Any]],
    base: EligibilityDecision,
    min_marker_hits: int = 1,
) -> EligibilityDecision:
    """
    Refine eligibility using DB chunk text (deterministic).

    Reason: retrieval can return high-scoring but semantically mismatched chunks for OOS terms;
    term-coverage against the top-1 chunk is a strong deterministic filter.
    """
    if not base.should_answer:
        return base

    topk = retrieval_trace.get("top_k_chunks") or []
    if not topk:
        return EligibilityDecision(False, "NO_RETRIEVAL")

    qn = normalize_for_matching(question_ar or "")
    looks_like_relationship = ("ما العلاقة" in qn) or ("العلاقة بين" in qn) or ("اربط بين" in qn) or ("ربط بين" in qn)
    looks_like_definition = ("عرّف" in qn) or ("تعريف" in qn) or ("كما ورد" in qn and "الاطار" in qn)

    top1_id = str((topk[0] or {}).get("chunk_id") or "").strip()
    if not top1_id:
        return EligibilityDecision(False, "NO_TOP1_CHUNK")

    row = (
        await session.execute(
            text("SELECT text_ar FROM chunk WHERE chunk_id=:cid"),
            {"cid": top1_id},
        )
    ).fetchone()
    if not row:
        return EligibilityDecision(False, "TOP1_CHUNK_NOT_FOUND")

    chunk_text = str(row.text_ar or "")

    # General term-coverage gate (OOS protection).
    # If we have no entity anchoring, require that at least a couple of content terms
    # from the question appear in the top-1 chunk text.
    # Reason: high-scoring retrieval can sometimes return generic chunks for unrelated queries.
    if not resolved_entities:
        q_words = [normalize_for_matching(w) for w in extract_arabic_words(question_ar or "")]
        stop = {
            "ما",
            "ماذا",
            "كيف",
            "هل",
            "لماذا",
            "متى",
            "اين",
            "أين",
            "الى",
            "إلى",
            "عن",
            "في",
            "من",
            "على",
            "هو",
            "هي",
            "هذا",
            "هذه",
            "ذلك",
            "تلك",
            "مع",
            "او",
            "أو",
            "ثم",
            "و",
        }
        content = sorted({w for w in q_words if w and (w not in stop) and len(w) >= 3})
        ev_norm = normalize_for_matching(chunk_text)
        hits = 0
        for w in content[:25]:
            if w in ev_norm:
                hits += 1
        if hits < 2:
            return EligibilityDecision(False, "LOW_TERM_OVERLAP_TOP1")

    # Relationship gate: require retrieval coverage for each resolved endpoint.
    # Reason: prevents answering when one endpoint is a spurious fuzzy match.
    if looks_like_relationship and len(resolved_entities) >= 2:
        chunk_ids = [str((t or {}).get("chunk_id") or "").strip() for t in topk[:10]]
        chunk_ids = [c for c in chunk_ids if c]
        if not chunk_ids:
            return EligibilityDecision(False, "NO_RETRIEVAL")

        # Map retrieved chunks to their entity (type,id).
        rows = (
            await session.execute(
                text(
                    """
                    SELECT chunk_id, entity_type, entity_id, chunk_type
                    FROM chunk
                    WHERE chunk_id = ANY(:cids)
                    """
                ),
                {"cids": chunk_ids},
            )
        ).fetchall()

        retrieved_entities = {(str(r.entity_type), str(r.entity_id)) for r in rows}
        resolved_pairs = {
            (str(e.get("type") or ""), str(e.get("id") or ""))
            for e in resolved_entities[:5]
            if (e.get("type") and e.get("id"))
        }

        # Require at least 2 resolved endpoints to be actually covered by retrieved chunks.
        covered = [p for p in sorted(resolved_pairs) if p in retrieved_entities]
        if len(covered) < 2:
            return EligibilityDecision(False, "MISSING_ENTITY_COVERAGE")

    # Definition gate: require a definition chunk for the resolved entity (or abstain).
    if looks_like_definition and resolved_entities:
        chunk_ids = [str((t or {}).get("chunk_id") or "").strip() for t in topk[:10]]
        chunk_ids = [c for c in chunk_ids if c]
        if chunk_ids:
            rows = (
                await session.execute(
                    text(
                        """
                        SELECT 1
                        FROM chunk
                        WHERE chunk_id = ANY(:cids)
                          AND chunk_type='definition'
                        LIMIT 1
                        """
                    ),
                    {"cids": chunk_ids},
                )
            ).fetchone()
            if not rows:
                return EligibilityDecision(False, "NO_DEFINITION_CHUNK")

    # Only enforce term-coverage for "OOS-attribute" style questions that can mention real entities
    # but ask for nonexistent attributes (e.g., color/code/degree/author/studies).
    marker_terms = [
        "لون",
        "كود",
        "الرقم",
        "رقم",
        "درجة",
        "مؤلف",
        "كاتب",
        "دراسة",
        "بحث",
        "تجربة",
        "قياس",
        "مؤشر",
    ]
    active_markers = [m for m in marker_terms if m in qn]
    if not active_markers:
        return base

    ev_norm = normalize_for_matching(chunk_text)
    hits = 0
    for m in active_markers[:10]:
        if normalize_for_matching(m) in ev_norm:
            hits += 1

    if hits < min_marker_hits:
        return EligibilityDecision(False, "LOW_MARKER_COVERAGE_TOP1")

    return base
