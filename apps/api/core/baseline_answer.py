"""
Baseline Answer Module

Provides a simple evidence-grounded answer without the Muḥāsibī reasoning structure.
Used for A/B testing to measure the value-add of the Muḥāsibī approach.
"""

from __future__ import annotations

from typing import Optional

from apps.api.core.schemas import (
    FinalResponse,
    Purpose,
    Citation,
    EntityRef,
    Difficulty,
    Confidence,
    EntityType,
)
from apps.api.retrieve.hybrid_retriever import HybridRetriever, RetrievalInputs


async def generate_baseline_answer(
    question: str,
    retriever: HybridRetriever,
    resolver,
    guardrails,
    language: str = "ar",
) -> FinalResponse:
    """
    Generate a baseline evidence-grounded answer.
    
    This skips the Muḥāsibī state machine and produces a minimal answer
    directly from retrieved evidence.
    
    Args:
        question: The user's question.
        retriever: The hybrid retriever with session attached.
        resolver: The entity resolver.
        guardrails: The guardrails instance.
        language: Response language preference.
        
    Returns:
        FinalResponse with minimal structure.
    """
    # Resolve entities
    resolved = resolver.resolve(question)
    resolved_list = [
        {"type": r.entity_type.value, "id": r.entity_id, "name_ar": r.name_ar}
        for r in resolved[:5]
    ]
    
    # Retrieve evidence
    session = getattr(retriever, "_session", None)
    inputs = RetrievalInputs(query=question, resolved_entities=resolved_list)
    
    if session:
        result = await retriever.retrieve(session, inputs)
    else:
        # No session, return not_found
        return _create_not_found_response(question)
    
    packets = result.evidence_packets
    
    # If no evidence, return not_found
    if not packets:
        return _create_not_found_response(question)
    
    # Build simple answer from evidence
    answer_parts = []
    citations = []
    entities_seen = set()
    entity_refs = []
    
    for packet in packets[:10]:  # Limit to top 10 packets
        # Add text to answer
        text = packet.get("text_ar", "")
        if text and len(answer_parts) < 5:
            # Truncate long texts
            if len(text) > 300:
                text = text[:300] + "..."
            answer_parts.append(text)
        
        # Add citation
        citations.append(Citation(
            chunk_id=packet.get("chunk_id", ""),
            source_anchor=packet.get("source_anchor", ""),
            ref=_get_first_ref(packet),
        ))
        
        # Add entity reference
        entity_key = (packet.get("entity_type"), packet.get("entity_id"))
        if entity_key not in entities_seen:
            entities_seen.add(entity_key)
            try:
                et = EntityType(packet.get("entity_type", "sub_value"))
            except ValueError:
                et = EntityType.SUB_VALUE
            entity_refs.append(EntityRef(
                type=et,
                id=packet.get("entity_id", ""),
                name_ar=packet.get("entity_name_ar", "") or packet.get("text_ar", "")[:30],
            ))
    
    # Combine answer
    answer_ar = "\n\n".join(answer_parts) if answer_parts else "لا توجد معلومات كافية."
    
    # Minimal Muḥāsibī-like structure (but simpler)
    return FinalResponse(
        listen_summary_ar=f"سؤال: {question}",
        purpose=Purpose(
            ultimate_goal_ar="الإجابة المباشرة من المصادر",
            constraints_ar=["استخدام الأدلة المتوفرة فقط"],
        ),
        path_plan_ar=["استرجاع الأدلة", "تجميع الإجابة"],
        answer_ar=answer_ar,
        citations=citations[:10],
        entities=entity_refs[:10],
        difficulty=Difficulty.MEDIUM,
        not_found=False,
        confidence=Confidence.MEDIUM if len(packets) >= 3 else Confidence.LOW,
    )


def _create_not_found_response(question: str) -> FinalResponse:
    """Create a not_found response."""
    return FinalResponse(
        listen_summary_ar=f"سؤال: {question}",
        purpose=Purpose(
            ultimate_goal_ar="الإجابة على السؤال",
            constraints_ar=["استخدام الأدلة المتوفرة فقط"],
        ),
        path_plan_ar=["البحث عن أدلة"],
        answer_ar="لم يتم العثور على معلومات كافية في قاعدة البيانات للإجابة على هذا السؤال.",
        citations=[],
        entities=[],
        difficulty=Difficulty.MEDIUM,
        not_found=True,
        confidence=Confidence.LOW,
    )


def _get_first_ref(packet: dict) -> Optional[str]:
    """Get the first reference string from a packet."""
    refs = packet.get("refs", [])
    if refs and isinstance(refs, list) and len(refs) > 0:
        ref = refs[0]
        if isinstance(ref, dict):
            return ref.get("ref", "")
        elif isinstance(ref, str):
            return ref
    return None
