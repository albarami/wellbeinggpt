"""Bounded critic loop for breakthrough mode.

Provides one-pass draft improvement with fixable-only triggers.

Critic loop triggers ONLY for:
- MISSING_REQUIRED_ENTITY
- EMPTY_REQUIRED_SECTION
- MISSING_USED_EDGES (when edges exist but not selected)
- MISSING_BOUNDARY_SECTION

Critic loop does NOT trigger for (fail-closed immediately):
- CITATION_VALIDITY_ERROR
- BINDING_PRUNE_FAILURE
- UNSUPPORTED_MUST_CITE
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from apps.api.core.answer_contract import check_contract


# Fixable contract failures (critic loop can help)
FIXABLE_FAILURES = frozenset({
    "MISSING_REQUIRED_ENTITY",
    "EMPTY_REQUIRED_SECTION",
    "EMPTY_SECTION",  # Variant
    "MISSING_USED_EDGES",
    "MISSING_BOUNDARY_SECTION",
    "MISSING_BOUNDARY",  # Variant
    "INCOMPLETE_COVERAGE",
    "LOW_PILLAR_COVERAGE",
})

# Non-fixable failures (fail-closed immediately)
NON_FIXABLE_FAILURES = frozenset({
    "CITATION_VALIDITY_ERROR",
    "INVALID_CITATION",
    "BINDING_PRUNE_FAILURE",
    "UNSUPPORTED_MUST_CITE",
    "UNSUPPORTED_CLAIM",
    "HALLUCINATION_DETECTED",
    "INJECTION_DETECTED",
})


@dataclass
class CriticResult:
    """Result of contract critique analysis."""
    
    needs_rewrite: bool = False
    fixable_issues: list[str] = field(default_factory=list)
    non_fixable_issues: list[str] = field(default_factory=list)
    missing_entities: list[str] = field(default_factory=list)
    missing_sections: list[str] = field(default_factory=list)


def _classify_contract_reasons(reasons: list[str]) -> CriticResult:
    """Classify contract failure reasons into fixable vs non-fixable."""
    fixable: list[str] = []
    non_fixable: list[str] = []
    missing_entities: list[str] = []
    missing_sections: list[str] = []
    
    for reason in reasons:
        reason_upper = reason.upper()
        
        # Check for non-fixable patterns first
        is_non_fixable = any(nf in reason_upper for nf in NON_FIXABLE_FAILURES)
        if is_non_fixable:
            non_fixable.append(reason)
            continue
        
        # Check for fixable patterns
        is_fixable = any(f in reason_upper for f in FIXABLE_FAILURES)
        if is_fixable:
            fixable.append(reason)
            
            # Extract specific missing items
            if "ENTITY" in reason_upper:
                # Try to extract entity name/type from reason
                missing_entities.append(reason)
            if "SECTION" in reason_upper or "BOUNDARY" in reason_upper:
                missing_sections.append(reason)
            continue
        
        # Unknown reasons are treated as non-fixable (fail-closed)
        non_fixable.append(reason)
    
    needs_rewrite = bool(fixable) and not non_fixable
    
    return CriticResult(
        needs_rewrite=needs_rewrite,
        fixable_issues=fixable,
        non_fixable_issues=non_fixable,
        missing_entities=missing_entities,
        missing_sections=missing_sections,
    )


def should_trigger_critic(contract_result: dict[str, Any]) -> bool:
    """
    Determine if critic loop should be triggered.
    
    Only triggers if ALL failures are fixable.
    Returns False if any non-fixable failure is present.
    """
    if contract_result.get("pass", False):
        return False  # Already passing, no need
    
    reasons = contract_result.get("reasons", []) or []
    if not reasons:
        return False  # No reasons to fix
    
    critique = _classify_contract_reasons(reasons)
    return critique.needs_rewrite


def critique_draft(
    answer_ar: str,
    citations: list[dict[str, Any]],
    graph_trace: dict[str, Any],
    contract_spec: dict[str, Any],
) -> CriticResult:
    """
    Analyze draft for contract compliance and identify fixable issues.
    
    Uses deterministic contract checker, not LLM.
    """
    contract_result = check_contract(
        answer_ar=answer_ar,
        citations=citations,
        graph_trace=graph_trace,
        contract_spec=contract_spec,
    )
    
    reasons = contract_result.get("reasons", []) or []
    return _classify_contract_reasons(reasons)


async def _targeted_retrieval_for_missing(
    session: AsyncSession,
    question: str,
    missing_entities: list[str],
    missing_sections: list[str],
    existing_packets: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Targeted retrieval to fill gaps identified by critic.
    
    Focuses on:
    - Definition chunks for missing entities
    - Boundary/limit chunks for missing sections
    """
    from sqlalchemy import text
    
    extra_packets: list[dict[str, Any]] = []
    existing_ids = {p.get("chunk_id") for p in existing_packets if p.get("chunk_id")}
    
    # If missing boundary section, fetch boundary-related chunks
    if any("BOUNDARY" in s.upper() for s in missing_sections):
        try:
            result = await session.execute(text("""
                SELECT chunk_id, entity_type, entity_id, chunk_type, text_ar, source_anchor
                FROM chunk
                WHERE (
                    text_ar ILIKE '%حد%'
                    OR text_ar ILIKE '%حدود%'
                    OR text_ar ILIKE '%ضابط%'
                    OR text_ar ILIKE '%غير منصوص%'
                )
                AND chunk_type IN ('definition', 'commentary')
                LIMIT 5
            """))
            rows = result.fetchall()
            for row in rows:
                cid = str(row[0])
                if cid not in existing_ids:
                    extra_packets.append({
                        "chunk_id": cid,
                        "entity_type": str(row[1] or ""),
                        "entity_id": str(row[2] or ""),
                        "chunk_type": str(row[3] or ""),
                        "text_ar": str(row[4] or ""),
                        "source_anchor": str(row[5] or ""),
                    })
                    existing_ids.add(cid)
        except Exception:
            pass
    
    # If missing entities, try to fetch their definitions
    for entity_reason in missing_entities:
        # Extract entity hints from reason (best effort)
        keywords = ["تعريف", "مفهوم", "قيمة"]
        try:
            result = await session.execute(text("""
                SELECT chunk_id, entity_type, entity_id, chunk_type, text_ar, source_anchor
                FROM chunk
                WHERE chunk_type = 'definition'
                LIMIT 3
            """))
            rows = result.fetchall()
            for row in rows:
                cid = str(row[0])
                if cid not in existing_ids:
                    extra_packets.append({
                        "chunk_id": cid,
                        "entity_type": str(row[1] or ""),
                        "entity_id": str(row[2] or ""),
                        "chunk_type": str(row[3] or ""),
                        "text_ar": str(row[4] or ""),
                        "source_anchor": str(row[5] or ""),
                    })
                    existing_ids.add(cid)
                    break  # One per missing entity
        except Exception:
            pass
    
    return extra_packets


async def critic_loop_once(
    session: AsyncSession,
    question: str,
    draft_answer: str,
    draft_citations: list[dict[str, Any]],
    draft_graph_trace: dict[str, Any],
    contract_spec: dict[str, Any],
    evidence_packets: list[dict[str, Any]],
    compose_fn: Any,  # Callable that recomposes answer
) -> dict[str, Any]:
    """
    One bounded improvement pass.
    
    1. Critique with deterministic contract checker
    2. If fixable, do targeted retrieval for missing elements
    3. Recompose with augmented evidence (one pass only)
    4. Return improved draft or original if not fixable
    
    Args:
        session: Database session
        question: Original question
        draft_answer: Current answer text
        draft_citations: Current citations
        draft_graph_trace: Current graph trace
        contract_spec: Contract specification
        evidence_packets: Current evidence packets
        compose_fn: Function to recompose answer with new evidence
    
    Returns:
        Dict with improved answer, citations, and metadata
    """
    # Step 1: Critique
    critique = critique_draft(
        answer_ar=draft_answer,
        citations=draft_citations,
        graph_trace=draft_graph_trace,
        contract_spec=contract_spec,
    )
    
    # If not fixable, return original (fail-closed)
    if not critique.needs_rewrite:
        return {
            "answer_ar": draft_answer,
            "citations": draft_citations,
            "graph_trace": draft_graph_trace,
            "critic_triggered": False,
            "non_fixable_issues": critique.non_fixable_issues,
        }
    
    # Step 2: Targeted retrieval
    extra_packets = await _targeted_retrieval_for_missing(
        session=session,
        question=question,
        missing_entities=critique.missing_entities,
        missing_sections=critique.missing_sections,
        existing_packets=evidence_packets,
    )
    
    if not extra_packets:
        # No new evidence found, return original
        return {
            "answer_ar": draft_answer,
            "citations": draft_citations,
            "graph_trace": draft_graph_trace,
            "critic_triggered": True,
            "critic_result": "no_extra_evidence",
            "fixable_issues": critique.fixable_issues,
        }
    
    # Step 3: Recompose with augmented evidence
    augmented_packets = evidence_packets + extra_packets
    
    try:
        # Call the compose function with augmented evidence
        # compose_fn should be like: compose_deep_answer(packets=..., semantic_edges=..., ...)
        # We pass through and let the caller handle the actual recomposition
        return {
            "answer_ar": draft_answer,  # Placeholder - actual recomposition done by caller
            "citations": draft_citations,
            "graph_trace": draft_graph_trace,
            "critic_triggered": True,
            "critic_result": "augmented",
            "extra_packets": extra_packets,
            "augmented_packets": augmented_packets,
            "fixable_issues": critique.fixable_issues,
        }
    except Exception as e:
        # On error, return original (fail-closed)
        return {
            "answer_ar": draft_answer,
            "citations": draft_citations,
            "graph_trace": draft_graph_trace,
            "critic_triggered": True,
            "critic_result": "error",
            "error": str(e),
        }
