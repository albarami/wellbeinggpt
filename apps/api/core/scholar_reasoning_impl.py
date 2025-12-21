"""Deterministic scholar reasoning implementation (deep + light deep).

Goal:
- Increase depth *safely* by expanding retrieval and producing a scholar-style structured answer.
- Preserve zero-hallucination: every substantive line is copied from stored evidence text.

This module intentionally avoids LLM synthesis. It constructs sections from:
- definition/evidence/commentary chunks
- grounded semantic edges (SCHOLAR_LINK + relation_type) with edge_justification_span quotes

If depth targets cannot be met without introducing uncited content, it abstains.

Reason: split implementation to keep modules <500 LOC.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from apps.api.core.schemas import Citation, Confidence
from apps.api.core.answer_contract import (
    UsedEdge,
    UsedEdgeSpan,
    build_argument_chains_from_used_edges,
    extract_compare_concepts_from_question,
)
from apps.api.core.scholar_reasoning_compose import compose_deep_answer, compose_light_answer
from apps.api.core.scholar_reasoning_compose_compare import compose_compare_answer
from apps.api.core.scholar_reasoning_compose_graph_intents import (
    compose_cross_pillar_path_answer,
    compose_network_answer,
    compose_tension_answer,
)
from apps.api.core.scholar_reasoning_compose_scenario import compose_partial_scenario_answer
from apps.api.core.scholar_reasoning_edge_fallback import semantic_edges_fallback
from apps.api.core.integrity_validator import validate_evidence_packets, get_integrity_warning_message
from apps.api.retrieve.graph_retriever import get_entity_neighbors
from apps.api.retrieve.hybrid_retriever import HybridRetriever, RetrievalInputs
from apps.api.retrieve.sql_retriever import get_chunks_with_refs, search_entities_by_name

# Breakthrough imports (optional, for enhanced reasoning)
import os
_ENABLE_BREAKTHROUGH = os.environ.get("MUHASIBI_ENABLE_BREAKTHROUGH", "").lower() in ("1", "true", "yes")


@dataclass(frozen=True)
class ScholarDepthTargets:
    """Depth floor targets for deep mode - MAXIMIZED FOR BRILLIANCE."""

    # Stage-2 deep targets (used for true deep questions) - INCREASED for depth
    min_claims_deep: int = 18
    # Light deep mode (used for many Gold QA questions) aims to improve rubric without bloating.
    min_claims_light: int = 12
    max_packets: int = 120  # More evidence for richer synthesis
    max_edges: int = 12  # More graph connections for depth


def _is_deep_question_ar(q_norm: str) -> bool:
    """
    Question-agnostic deep classifier.
    
    Reason: ANY substantive wellbeing question deserves deep, scholarly treatment.
    We don't gate on specific keywords - if it's a real question, give it depth.
    
    Only exclude trivial/structural questions (single word, greetings, etc).
    """
    q = (q_norm or "").strip()
    
    # Trivial questions - don't need deep mode
    if len(q) < 15:
        return False
    
    # Structural/navigation questions - handled separately
    structural_markers = [
        "ما هي ركائز",
        "ما هي الركائز",
        "اذكر الركائز",
        "عدد الركائز",
        "كم ركيزة",
        "قائمة القيم",
    ]
    if any(m in q for m in structural_markers):
        return False
    
    # Everything else gets deep treatment - question agnostic
    return True


def _is_light_deep_question_ar(q_norm: str) -> bool:
    """
    Light deep classifier for shorter questions.
    
    This is now a fallback - most questions go through full deep mode.
    Light mode is for very short definition-style questions.
    """
    q = (q_norm or "").strip()
    
    # Very short questions that are still substantive
    # (between 15-50 chars - single concept questions)
    if 15 <= len(q) <= 50:
        return True
    
    return False


def _intent_type_ar(q_norm: str, detected_pillars: list[str] | None = None) -> str:
    """
    Question-agnostic intent classifier.
    
    Reason: Don't rely on specific keywords. Instead:
    1. Use detected entities to determine if cross-pillar reasoning is needed
    2. Use question structure (multiple parts, "و", enumeration) for compare
    3. Default to "generic" which still gets full scholar treatment
    
    The evidence and graph will guide the answer, not hardcoded keywords.
    
    Note: global_synthesis intent uses keyword override per adjustment #6.
    Patterns are pre-normalized for matching against normalized question text.
    """
    from apps.api.retrieve.normalize_ar import normalize_for_matching
    
    q = (q_norm or "").strip()
    pillars = detected_pillars or []
    
    # GLOBAL_SYNTHESIS keyword patterns (20 paraphrases per adjustment #6)
    # Patterns must be normalized to match normalized question text
    # Use keyword override as secondary signal, but check early for performance
    GLOBAL_SYNTHESIS_PATTERNS_RAW = [
        # Arabic - flourishing/wellbeing
        "ازدهار", "حياة طيبة", "الحياة الطيبة", "كيف يسهم الإطار",
        "البشرية", "المجتمع", "نهضة المجتمع", "خير البشرية",
        "رفاهية الإنسان", "سعادة الإنسان", "الإنسان الكامل",
        # Arabic - integration/holistic
        "التكامل الشامل", "المنظومة الكاملة", "الرؤية الشاملة",
        "كيف تعمل الركائز معا", "العلاقة بين جميع الركائز",
        "الصورة الكبرى", "المنظور الكلي", "الإطار ككل",
        # English
        "flourishing", "human wellbeing",
    ]
    
    # Normalize patterns for comparison
    GLOBAL_SYNTHESIS_PATTERNS = [normalize_for_matching(p) for p in GLOBAL_SYNTHESIS_PATTERNS_RAW]
    
    if any(p in q for p in GLOBAL_SYNTHESIS_PATTERNS if p):
        return "global_synthesis"
    
    # If multiple pillars detected in question → cross-pillar reasoning
    if len(pillars) >= 2:
        # Check if it's asking for a path/sequence
        if any(x in q for x in ["→", "ثم", "إلى", "الى", "من"]):
            return "cross_pillar_path"
        return "cross_pillar"
    
    # Explicit compare structure (multiple items with "و" or "أو")
    # Count quoted items or items separated by و
    quote_count = q.count('"') + q.count('"') + q.count('"')
    if quote_count >= 4:  # At least 2 quoted concepts
        if any(x in q for x in ["فرق", "قارن", "مقارنة", "يختلف"]):
            return "compare"
    
    # Explicit tension/reconciliation
    if any(x in q for x in ["تعارض", "توفيق", "نجمع بين"]):
        return "tension"
    
    # Scenario/case analysis
    if any(x in q for x in ["حالة", "سيناريو", "موقف"]):
        return "scenario"
    
    # Network building (explicit request)
    if any(x in q for x in ["شبكة", "ثلاث ركائز", "ثلاثة"]):
        return "network"
    
    # Default: let evidence guide the answer
    # Generic still gets full scholar treatment with edges if found
    return "generic"


def _dedupe_citations(citations: list[Citation]) -> list[Citation]:
    """
    Deduplicate citations by chunk_id.
    
    Note: For UI evidence panel, additional deduplication by 
    (chunk_id, span_start, span_end) and normalized quote is done in ui.py.
    """
    seen: set[str] = set()
    out: list[Citation] = []
    for c in citations:
        k = str(c.chunk_id)
        if not k or k in seen:
            continue
        seen.add(k)
        out.append(c)
    return out


def _count_claim_lines(answer_ar: str) -> int:
    """Count bullet-like lines (our proxy for must-cite claim count)."""

    lines = [ln.strip() for ln in (answer_ar or "").splitlines() if ln.strip()]
    return sum(1 for ln in lines if ln.startswith("-"))



class ScholarReasoner:
    """Deterministic deep-mode answer generator.
    
    With enable_breakthrough=True, uses:
    - Candidate generation + ranking for optimal edge selection
    - Bounded critic loop for one-pass improvement
    """

    def __init__(
        self,
        *,
        session: AsyncSession,
        retriever: HybridRetriever,
        targets: ScholarDepthTargets = ScholarDepthTargets(),
        enable_breakthrough: bool | None = None,
    ):
        self.session = session
        self.retriever = retriever
        self.targets = targets
        # Use env var if not explicitly set
        self.enable_breakthrough = enable_breakthrough if enable_breakthrough is not None else _ENABLE_BREAKTHROUGH

    async def generate(
        self,
        *,
        question: str,
        question_norm: str,
        detected_entities: list[dict[str, Any]],
        evidence_packets: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Generate a deep scholar answer from evidence only."""

        is_deep = _is_deep_question_ar(question_norm)
        is_light = (not is_deep) and _is_light_deep_question_ar(question_norm)
        if not (is_deep or is_light):
            return {"answer_ar": "", "citations": [], "not_found": False, "confidence": Confidence.MEDIUM}

        expanded_packets = await self._expand_retrieval(question=question, detected_entities=detected_entities)
        packets = expanded_packets or list(evidence_packets or [])

        # INTEGRITY VALIDATION: Filter quarantined chunks and flag data quality issues
        # Reason: A scholar is a critic of his own sources - don't propagate malformed evidence
        packets, integrity_results = validate_evidence_packets(packets)
        integrity_warning = get_integrity_warning_message(integrity_results)
        # Note: integrity_warning can be surfaced in UI metadata if needed

        semantic_edges: list[dict[str, Any]] = []
        if is_deep:
            semantic_edges = await self._semantic_edges_from_entities(detected_entities)
            if not semantic_edges:
                semantic_edges = await semantic_edges_fallback(
                    session=self.session,
                    question_norm=question_norm,
                    max_edges=self.targets.max_edges,
                )
            
            # BREAKTHROUGH: Use candidate ranking for optimal edge selection
            if self.enable_breakthrough and semantic_edges:
                try:
                    from apps.api.core.breakthrough.candidate_generator import generate_candidates_sync
                    from apps.api.core.breakthrough.candidate_ranker import pick_best
                    
                    detected_pillar_ids = [
                        str(e.get("entity_id") or "")
                        for e in (detected_entities or [])
                        if str(e.get("entity_type") or "") == "pillar"
                    ]
                    
                    candidates = generate_candidates_sync(
                        edges=semantic_edges,
                        intent=_intent_type_ar(question_norm, detected_pillar_ids),
                        detected_pillar_ids=detected_pillar_ids if detected_pillar_ids else None,
                        max_candidates=5,
                        max_edges_per_candidate=self.targets.max_edges,
                    )
                    
                    if candidates:
                        best = pick_best(candidates)
                        if best and best.used_edges:
                            semantic_edges = best.used_edges
                except Exception:
                    pass  # Fall back to original edges on error
            
            if semantic_edges:
                packets = await self._expand_packets_from_edges(packets, semantic_edges)

        used_edges: list[dict[str, Any]] = []
        if is_deep:
            # Extract pillar IDs from detected entities for question-agnostic intent classification
            detected_pillar_ids = [
                str(e.get("entity_id") or "")
                for e in (detected_entities or [])
                if str(e.get("entity_type") or "") == "pillar"
            ]
            intent = _intent_type_ar(question_norm, detected_pillar_ids)
            
            # BRILLIANCE RULE: If we have evidence, we MUST synthesize, NEVER refuse.
            # When a concept isn't in framework, contrast it with what IS in framework.
            
            if intent == "compare":
                concepts = list(extract_compare_concepts_from_question(question) or [])
                # Try to include the defined concept if explicitly mentioned.
                if "عرّف" in (question_norm or "") or "عرف" in (question_norm or ""):
                    _, _, tail = (question or "").partition("عرّف")
                    if not tail:
                        _, _, tail = (question or "").partition("عرف")
                    tok = (tail.strip().split() or [""])[0]
                    tok = tok.strip("،.؛:()[]\"'«»")
                    if tok in {"داخل", "هذا", "الإطار", "في"} and len((tail.strip().split() or [])) > 1:
                        tok = (tail.strip().split() or [""])[1].strip("،.؛:()[]\"'«»")
                    head = tok.strip()
                    if head and head not in concepts:
                        concepts = [head] + concepts

                # Check which concepts actually exist in framework
                concepts_found: list[str] = []
                concepts_external: list[str] = []
                try:
                    from apps.api.core.schemas import EntityType
                    extra_packets: list[dict[str, Any]] = []
                    
                    for concept in concepts[:6]:
                        nm = str(concept or "").strip()
                        if not nm:
                            continue
                        hits = await search_entities_by_name(self.session, name_pattern=nm, limit=4)
                        if hits:
                            concepts_found.append(nm)
                            for h in hits[:2]:
                                et = str(h.get("entity_type") or "")
                                eid = str(h.get("id") or "")
                                if not (et and eid):
                                    continue
                                if et not in {"pillar", "core_value", "sub_value"}:
                                    continue
                                try:
                                    etype = EntityType(et)
                                except Exception:
                                    continue
                                extra_packets.extend(await get_chunks_with_refs(self.session, etype, eid, limit=10))
                        else:
                            concepts_external.append(nm)
                    
                    if extra_packets:
                        seen = {str(p.get("chunk_id") or "") for p in packets}
                        for p in extra_packets:
                            cid = str(p.get("chunk_id") or "")
                            if cid and cid not in seen:
                                packets.append(p)
                                seen.add(cid)
                except Exception:
                    pass
                
                # BRILLIANCE: If some concepts are external (not in framework), use contrast mode
                # This handles questions like "What distinguishes this framework from X?"
                if concepts_external and packets:
                    # Fall through to deep synthesis - explain framework's richness
                    # by contrasting with the external concept's limitations
                    answer_ar, citations, used_edges = compose_deep_answer(
                        packets=packets,
                        semantic_edges=semantic_edges,
                        max_edges=self.targets.max_edges,
                        question_ar=question,
                        prefer_more_claims=True,  # Maximum depth for contrast
                    )
                elif concepts_found:
                    # BRILLIANCE ENHANCEMENT: Also find edges between compared concepts
                    # Even intra-pillar concepts can have SCHOLAR_LINK edges
                    compare_edges: list[dict[str, Any]] = []
                    try:
                        from apps.api.retrieve.graph_retriever import get_entity_neighbors
                        for concept in concepts_found[:4]:
                            hits = await search_entities_by_name(self.session, name_pattern=concept, limit=2)
                            for h in hits[:1]:
                                et = str(h.get("entity_type") or "")
                                eid = str(h.get("id") or "")
                                if et and eid:
                                    neigh = await get_entity_neighbors(
                                        self.session, et, eid,
                                        relationship_types=["SCHOLAR_LINK"],
                                        direction="both",
                                        status="approved",
                                    )
                                    for n in neigh[:3]:
                                        if n.get("justification_spans"):
                                            n["source_type"] = et
                                            n["source_id"] = eid
                                            compare_edges.append(n)
                    except Exception:
                        pass
                    
                    # Use deep_answer for richer output when we have edges
                    if compare_edges:
                        answer_ar, citations, used_edges = compose_deep_answer(
                            packets=packets,
                            semantic_edges=compare_edges,
                            max_edges=8,
                            question_ar=question,
                            prefer_more_claims=True,  # Maximum brilliance
                        )
                    else:
                        answer_ar, citations, used_edges = compose_compare_answer(
                            question_ar=question,
                            concepts_ar=concepts_found,
                            packets=packets,
                            prefer_more_claims=True,  # More depth even without edges
                        )
                else:
                    # No concepts found but we have evidence - synthesize brilliantly
                    answer_ar, citations, used_edges = compose_deep_answer(
                        packets=packets,
                        semantic_edges=semantic_edges,
                        max_edges=self.targets.max_edges,
                        question_ar=question,
                        prefer_more_claims=True,
                    )
            elif intent == "network":
                answer_ar, citations, used_edges = compose_network_answer(
                    packets=packets,
                    semantic_edges=semantic_edges,
                    max_links=5,
                    query=question,
                )
            elif intent == "tension":
                answer_ar, citations, used_edges = compose_tension_answer(
                    packets=packets,
                    semantic_edges=semantic_edges,
                )
            elif intent == "scenario":
                answer_ar, citations, used_edges = compose_partial_scenario_answer(
                    packets=packets,
                    question_ar=question,
                    prefer_more_claims=False,
                )
            elif intent == "global_synthesis":
                # World Model global synthesis - uses loops, interventions, simulations
                try:
                    from apps.api.core.world_model.cache import get_cached_loops
                    from apps.api.core.world_model.loop_reasoner import retrieve_relevant_loops
                    from apps.api.core.world_model.intervention_planner import compute_intervention_plan
                    from apps.api.core.world_model.simulator import simulate_change
                    from apps.api.core.world_model.composer import (
                        build_world_model_plan,
                        compose_global_synthesis_answer,
                    )
                    
                    # Detect loops
                    all_loops = await get_cached_loops(self.session)
                    relevant_loops = retrieve_relevant_loops(
                        all_loops, detected_entities, detected_pillar_ids, top_k=5
                    )
                    
                    # Compute intervention plan
                    goal_ar = question  # Use question as goal
                    intervention = await compute_intervention_plan(
                        self.session, goal_ar, detected_entities, relevant_loops, max_steps=7
                    )
                    
                    # Build plan
                    plan = build_world_model_plan(
                        loops=relevant_loops,
                        interventions=[intervention] if intervention.steps else [],
                        simulations=[],
                        detected_pillars=detected_pillar_ids,
                    )
                    
                    # Compose answer
                    answer_ar, citations, used_edges = compose_global_synthesis_answer(
                        plan=plan,
                        question_ar=question,
                        packets=packets,
                    )
                except Exception:
                    # Fallback to deep answer if world model fails
                    answer_ar, citations, used_edges = compose_deep_answer(
                        packets=packets,
                        semantic_edges=semantic_edges,
                        max_edges=self.targets.max_edges,
                        question_ar=question,
                        prefer_more_claims=True,
                    )
            elif intent in {"cross_pillar_path"}:
                answer_ar, citations, used_edges = compose_cross_pillar_path_answer(
                    packets=packets,
                    semantic_edges=semantic_edges,
                    max_steps=5,
                    prefer_more_claims=False,
                )
            else:
                answer_ar, citations, used_edges = compose_deep_answer(
                    packets=packets,
                    semantic_edges=semantic_edges,
                    max_edges=self.targets.max_edges,
                    question_ar=question,
                    prefer_more_claims=False,
                )
            # Intent-specific depth floors:
            # - Graph intents are structured and typically shorter; require a smaller floor to avoid false "not_found".
            # - Compare answers are matrices; also smaller than full deep narrative.
            # - Global synthesis has lower floor since it's structured with loops/interventions.
            if intent in {"cross_pillar_path", "network", "tension", "global_synthesis"}:
                min_claims = 6
            elif intent == "compare":
                min_claims = 10
            else:
                min_claims = int(self.targets.min_claims_deep)
        else:
            answer_ar, citations = compose_light_answer(packets=packets, prefer_more_claims=False)
            min_claims = int(self.targets.min_claims_light)

        claim_lines = _count_claim_lines(answer_ar)
        if claim_lines < min_claims:
            packets2 = await self._expand_retrieval(question=question, detected_entities=detected_entities, extra=True)
            packets2 = packets2 or packets

            if is_deep:
                semantic_edges2 = await self._semantic_edges_from_entities(detected_entities, extra=True)
                if not semantic_edges2:
                    semantic_edges2 = await semantic_edges_fallback(
                        session=self.session,
                        question_norm=question_norm,
                        max_edges=(self.targets.max_edges + 4),
                        extra=True,
                    )
                if semantic_edges2:
                    packets2 = await self._expand_packets_from_edges(packets2, semantic_edges2, extra=True)
                # Retry with more claims: keep same intent specialization but allow more edges/spans.
                intent = _intent_type_ar(question_norm, detected_pillar_ids)
                if intent == "compare":
                    concepts = list(extract_compare_concepts_from_question(question) or [])
                    answer_ar, citations, used_edges = compose_compare_answer(
                        question_ar=question,
                        concepts_ar=concepts,
                        packets=packets2,
                        prefer_more_claims=True,
                    )
                elif intent == "network":
                    answer_ar, citations, used_edges = compose_network_answer(
                        packets=packets2,
                        semantic_edges=semantic_edges2,
                        max_links=6,
                        query=question,
                    )
                elif intent == "tension":
                    answer_ar, citations, used_edges = compose_tension_answer(
                        packets=packets2,
                        semantic_edges=semantic_edges2,
                    )
                elif intent == "cross_pillar_path":
                    answer_ar, citations, used_edges = compose_cross_pillar_path_answer(
                        packets=packets2,
                        semantic_edges=semantic_edges2,
                        max_steps=4,
                        prefer_more_claims=True,
                    )
                elif intent == "scenario":
                    answer_ar, citations, used_edges = compose_partial_scenario_answer(
                        packets=packets2,
                        question_ar=question,
                        prefer_more_claims=True,
                    )
                elif intent == "global_synthesis":
                    # Retry global synthesis with expanded evidence
                    try:
                        from apps.api.core.world_model.cache import get_cached_loops
                        from apps.api.core.world_model.loop_reasoner import retrieve_relevant_loops
                        from apps.api.core.world_model.intervention_planner import compute_intervention_plan
                        from apps.api.core.world_model.composer import (
                            build_world_model_plan,
                            compose_global_synthesis_answer,
                        )
                        
                        all_loops = await get_cached_loops(self.session)
                        relevant_loops = retrieve_relevant_loops(
                            all_loops, detected_entities, detected_pillar_ids, top_k=5
                        )
                        intervention = await compute_intervention_plan(
                            self.session, question, detected_entities, relevant_loops, max_steps=7
                        )
                        plan = build_world_model_plan(
                            loops=relevant_loops,
                            interventions=[intervention] if intervention.steps else [],
                            simulations=[],
                            detected_pillars=detected_pillar_ids,
                        )
                        answer_ar, citations, used_edges = compose_global_synthesis_answer(
                            plan=plan,
                            question_ar=question,
                            packets=packets2,
                        )
                    except Exception:
                        answer_ar, citations, used_edges = compose_deep_answer(
                            packets=packets2,
                            semantic_edges=semantic_edges2,
                            max_edges=self.targets.max_edges,
                            question_ar=question,
                            prefer_more_claims=True,
                        )
                else:
                    answer_ar, citations, used_edges = compose_deep_answer(
                        packets=packets2,
                        semantic_edges=semantic_edges2,
                        max_edges=self.targets.max_edges,
                        question_ar=question,
                        prefer_more_claims=True,
                    )
            else:
                answer_ar, citations = compose_light_answer(packets=packets2, prefer_more_claims=True)

        claim_lines = _count_claim_lines(answer_ar)
        citations = _dedupe_citations(citations)

        # BRILLIANCE RULE: If we have evidence packets, NEVER refuse.
        # Instead, do a last-resort brilliant synthesis.
        if (claim_lines < min_claims or not citations) and packets:
            # Force a generous deep synthesis - use ALL available evidence
            answer_ar, citations, used_edges = compose_deep_answer(
                packets=packets,
                semantic_edges=semantic_edges if is_deep else [],
                max_edges=12,  # Maximum edges for depth
                question_ar=question,
                prefer_more_claims=True,
            )
            citations = _dedupe_citations(citations)
        
        # Only refuse if we truly have NO evidence at all
        if not citations and not packets:
            return {
                "answer_ar": "لا يوجد في البيانات الحالية ما يدعم الإجابة على هذا السؤال.",
                "citations": [],
                "not_found": True,
                "confidence": Confidence.LOW,
            }

        # BREAKTHROUGH: Bounded critic loop for fixable contract failures
        if self.enable_breakthrough and is_deep:
            try:
                from apps.api.core.breakthrough.critic_loop import (
                    should_trigger_critic,
                    critic_loop_once,
                )
                from apps.api.core.answer_contract import check_contract
                
                # Build a minimal contract spec for the question
                contract_spec = {
                    "required_entities": detected_pillar_ids[:3] if detected_pillar_ids else [],
                    "graph_required": bool(semantic_edges),
                    "boundary_required": "حدود" in (question_norm or "") or "ضوابط" in (question_norm or ""),
                }
                
                # Check current answer against contract
                contract_result = check_contract(
                    answer_ar=answer_ar,
                    citations=[c.model_dump() if hasattr(c, "model_dump") else c for c in citations],
                    graph_trace={"used_edges": used_edges},
                    contract_spec=contract_spec,
                )
                
                # Run critic loop ONLY if failures are fixable
                if should_trigger_critic(contract_result):
                    critic_result = await critic_loop_once(
                        session=self.session,
                        question=question,
                        draft_answer=answer_ar,
                        draft_citations=[c.model_dump() if hasattr(c, "model_dump") else c for c in citations],
                        draft_graph_trace={"used_edges": used_edges},
                        contract_spec=contract_spec,
                        evidence_packets=packets,
                        compose_fn=None,  # Not using recompose in this version
                    )
                    
                    # If critic found extra evidence, merge it
                    if critic_result.get("extra_packets"):
                        extra = critic_result["extra_packets"]
                        packets_merged = packets + extra
                        # Recompose with augmented evidence
                        if intent in {"compare"}:
                            concepts = list(extract_compare_concepts_from_question(question) or [])
                            answer_ar, citations, used_edges = compose_compare_answer(
                                question_ar=question,
                                concepts_ar=concepts,
                                packets=packets_merged,
                                prefer_more_claims=True,
                            )
                        else:
                            answer_ar, citations, used_edges = compose_deep_answer(
                                packets=packets_merged,
                                semantic_edges=semantic_edges,
                                max_edges=self.targets.max_edges,
                                question_ar=question,
                                prefer_more_claims=True,
                            )
                        citations = _dedupe_citations(citations)
            except Exception:
                pass  # Fall back to original answer on error

        return {
            "answer_ar": answer_ar,
            "citations": citations,
            "used_edges": used_edges,
            "argument_chains": [
                {
                    "edge_id": ac.edge_id,
                    "relation_type": ac.relation_type,
                    "from_node": ac.from_node,
                    "to_node": ac.to_node,
                    "claim_ar": ac.claim_ar,
                    "inference_type": ac.inference_type,
                    "evidence_spans": [
                        {
                            "chunk_id": sp.chunk_id,
                            "span_start": int(sp.span_start),
                            "span_end": int(sp.span_end),
                            "quote": sp.quote,
                        }
                        for sp in (ac.evidence_spans or ())
                    ],
                    "boundary_ar": ac.boundary_ar,
                    "boundary_spans": [
                        {
                            "chunk_id": sp.chunk_id,
                            "span_start": int(sp.span_start),
                            "span_end": int(sp.span_end),
                            "quote": sp.quote,
                        }
                        for sp in (ac.boundary_spans or ())
                    ],
                }
                for ac in build_argument_chains_from_used_edges(
                    used_edges=[
                        UsedEdge(
                            edge_id=str(ue.get("edge_id") or ""),
                            from_node=str(ue.get("from_node") or ""),
                            to_node=str(ue.get("to_node") or ""),
                            relation_type=str(ue.get("relation_type") or ""),
                            justification_spans=tuple(
                                [
                                    UsedEdgeSpan(
                                        chunk_id=str(sp.get("chunk_id") or ""),
                                        span_start=int(sp.get("span_start") or 0),
                                        span_end=int(sp.get("span_end") or 0),
                                        quote=str(sp.get("quote") or ""),
                                    )
                                    for sp in list(ue.get("justification_spans") or [])[:6]
                                    if str(sp.get("chunk_id") or "")
                                ]
                            ),
                        )
                        for ue in list(used_edges or [])[:16]
                    ]
                )
            ],
            "not_found": False,
            "confidence": Confidence.MEDIUM,
        }

    async def _expand_retrieval(
        self,
        *,
        question: str,
        detected_entities: list[dict[str, Any]],
        extra: bool = False,
    ) -> list[dict[str, Any]]:
        """Re-run retrieval with larger caps (depth expansion)."""

        try:
            old_max = int(getattr(getattr(self.retriever, "merge_ranker", None), "max_packets", 25) or 25)
        except Exception:
            old_max = 25

        target_max = min(140, self.targets.max_packets + (30 if extra else 0))
        try:
            self.retriever.merge_ranker.max_packets = target_max  # type: ignore[attr-defined]
        except Exception:
            pass

        try:
            resolved = []
            for e in (detected_entities or [])[:6]:
                if e.get("type") and e.get("id"):
                    resolved.append(
                        {"type": e["type"], "id": str(e["id"]), "confidence": float(e.get("confidence") or 0.0)}
                    )
            inputs = RetrievalInputs(query=question, resolved_entities=resolved, top_k=35, graph_depth=3)
            merged = await self.retriever.retrieve(self.session, inputs)
            return list(merged.evidence_packets or [])
        except Exception:
            return []
        finally:
            try:
                self.retriever.merge_ranker.max_packets = old_max  # type: ignore[attr-defined]
            except Exception:
                pass

    async def _semantic_edges_from_entities(
        self,
        detected_entities: list[dict[str, Any]],
        extra: bool = False,
    ) -> list[dict[str, Any]]:
        """Fetch grounded semantic edges (SCHOLAR_LINK) for anchor entities."""

        edges_out: list[dict[str, Any]] = []
        max_edges = self.targets.max_edges + (4 if extra else 0)
        min_strength = 0.2 if extra else 0.4

        for e in (detected_entities or [])[:3]:
            et = str(e.get("type") or "")
            eid = str(e.get("id") or "")
            if not et or not eid:
                continue
            neigh = await get_entity_neighbors(
                self.session,
                et,
                eid,
                relationship_types=["SCHOLAR_LINK"],
                direction="both",
                status="approved",
            )
            for n in neigh:
                # Annotate source endpoint for downstream used-edge tracing.
                n["source_type"] = et
                n["source_id"] = eid
                rt = (n.get("relation_type") or "")
                spans = n.get("justification_spans") or []
                try:
                    strength = float(n.get("strength_score") or 0.0)
                except Exception:
                    strength = 0.0
                if rt and spans and (strength >= min_strength):
                    edges_out.append(n)
                if len(edges_out) >= max_edges:
                    break
            if len(edges_out) >= max_edges:
                break

        return edges_out

    async def _expand_packets_from_edges(
        self,
        packets: list[dict[str, Any]],
        semantic_edges: list[dict[str, Any]],
        extra: bool = False,
    ) -> list[dict[str, Any]]:
        """Pull chunks for linked nodes so deep answers have more grounded material."""

        out = list(packets or [])
        seen = {str(p.get("chunk_id")) for p in out if p.get("chunk_id")}

        per_neighbor_notes = 2
        per_neighbor_other = 2 if extra else 1

        for e in semantic_edges[: self.targets.max_edges]:
            n_type = str(e.get("neighbor_type") or "")
            n_id = str(e.get("neighbor_id") or "")
            if not n_type or not n_id:
                continue
            try:
                from apps.api.core.schemas import EntityType

                et = EntityType(n_type)
            except Exception:
                continue

            more = await get_chunks_with_refs(self.session, et, n_id, limit=12)
            notes = [p for p in more if str(p.get("chunk_id") or "").startswith("SN_")]
            other = [p for p in more if not str(p.get("chunk_id") or "").startswith("SN_")]
            chosen = notes[:per_neighbor_notes] + other[:per_neighbor_other]
            for p in chosen:
                cid = str(p.get("chunk_id") or "")
                if cid and cid not in seen:
                    out.append(p)
                    seen.add(cid)

        return out

