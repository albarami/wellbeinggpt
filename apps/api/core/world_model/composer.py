"""Answer composition for World Model global synthesis.

This module composes structured answers for GLOBAL_SYNTHESIS intent:
- Uses detected loops + interventions + mechanisms
- Covers all 5 pillars when possible
- Generates evidence-bound prose with citations

Pipeline:
1. DraftWriter: Natural prose structure
2. WorldModel Plan Builder: Select relevant loops + interventions
3. Writer: Compose narrative using the plan
4. Evidence binding from spans
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from apps.api.core.schemas import Citation
from apps.api.core.world_model.schemas import (
    DetectedLoop,
    InterventionPlan,
    SimulationResult,
)


@dataclass
class WorldModelPlan:
    """Plan for world model answer composition."""
    loops: list[DetectedLoop] = field(default_factory=list)
    interventions: list[InterventionPlan] = field(default_factory=list)
    simulations: list[SimulationResult] = field(default_factory=list)
    covered_pillars: set[str] = field(default_factory=set)
    used_edges: list[dict[str, Any]] = field(default_factory=list)


def _extract_pillar_from_node(node_ref: str) -> str | None:
    """Extract pillar ID from node reference."""
    if not node_ref:
        return None
    if "pillar:" in node_ref:
        return node_ref.split("pillar:")[1].split(":")[0].split()[0]
    # Check for pillar ID pattern
    parts = node_ref.split(":")
    if len(parts) >= 2:
        ref_id = parts[1]
        if ref_id.startswith("P00"):
            return ref_id
    return None


def build_world_model_plan(
    *,
    loops: list[DetectedLoop],
    interventions: list[InterventionPlan],
    simulations: list[SimulationResult],
    detected_pillars: list[str],
    target_loop_count: int = 2,
    target_intervention_count: int = 1,
) -> WorldModelPlan:
    """Build a plan for world model answer composition.
    
    Selects the most relevant loops and interventions while maximizing
    pillar coverage.
    
    Args:
        loops: All detected loops
        interventions: Available intervention plans
        simulations: Available simulation results
        detected_pillars: Pillars detected in the question
        target_loop_count: Target number of loops to include
        target_intervention_count: Target number of interventions
        
    Returns:
        WorldModelPlan with selected components
    """
    plan = WorldModelPlan()
    
    # Track pillar coverage
    covered: set[str] = set()
    
    # Select loops (prioritize those covering detected pillars)
    sorted_loops = sorted(
        loops,
        key=lambda l: sum(1 for p in detected_pillars if any(p in n for n in l.nodes)),
        reverse=True,
    )
    
    for loop in sorted_loops[:target_loop_count]:
        plan.loops.append(loop)
        
        # Track covered pillars
        for node in loop.nodes:
            pillar = _extract_pillar_from_node(node)
            if pillar:
                covered.add(pillar)
        
        # Build used_edges from loop
        for edge_id in loop.edge_ids:
            # Find matching evidence spans
            edge_spans = []
            for span in loop.evidence_spans:
                if isinstance(span, dict):
                    edge_spans.append(span)
            
            plan.used_edges.append({
                "edge_id": edge_id,
                "from_node": loop.nodes[0] if loop.nodes else "",
                "to_node": loop.nodes[1] if len(loop.nodes) > 1 else "",
                "relation_type": "LOOP_MEMBER",
                "justification_spans": edge_spans[:2],
            })
    
    # Select interventions
    for intervention in interventions[:target_intervention_count]:
        plan.interventions.append(intervention)
        
        # Track covered pillars from intervention steps
        for step in intervention.steps:
            if step.target_node_ref_kind == "pillar":
                covered.add(step.target_node_ref_id)
    
    # Add simulations
    for sim in simulations[:1]:
        plan.simulations.append(sim)
    
    plan.covered_pillars = covered
    
    return plan


def _format_loop_section(loop: DetectedLoop) -> tuple[str, list[Citation]]:
    """Format a loop for the answer.
    
    Returns:
        Tuple of (formatted text, citations)
    """
    citations: list[Citation] = []
    lines: list[str] = []
    
    # Header
    loop_type_ar = "تعزيزية" if loop.loop_type == "reinforcing" else "موازنة"
    lines.append(f"- **حلقة {loop_type_ar}**: {' ← '.join(loop.node_labels_ar[:4])}")
    
    # Evidence
    summary = loop.generate_summary_ar()
    if summary and summary != "غير منصوص عليه في الإطار":
        lines.append(f"  - الدليل: {summary[:200]}")
        
        # Extract citations from spans
        for span in loop.evidence_spans[:2]:
            if isinstance(span, dict) and span.get("chunk_id"):
                citations.append(Citation(
                    chunk_id=str(span["chunk_id"]),
                    source_anchor=str(span.get("span_start", 0)),
                    ref=None,
                ))
    
    return "\n".join(lines), citations


def _format_intervention_section(plan: InterventionPlan) -> tuple[str, list[Citation]]:
    """Format an intervention plan for the answer.
    
    Returns:
        Tuple of (formatted text, citations)
    """
    citations: list[Citation] = []
    lines: list[str] = []
    
    lines.append(f"- **الهدف**: {plan.goal_ar}")
    lines.append("- **خطوات التدخل**:")
    
    for i, step in enumerate(plan.steps[:5], 1):
        lines.append(f"  {i}. {step.target_node_label_ar}")
        if step.mechanism_reason_ar and step.mechanism_reason_ar != "غير منصوص":
            reason = step.mechanism_reason_ar[:100]
            lines.append(f"     - السبب: {reason}")
        
        # Extract citations
        for cit in step.mechanism_citations[:1]:
            if isinstance(cit, dict) and cit.get("chunk_id"):
                citations.append(Citation(
                    chunk_id=str(cit["chunk_id"]),
                    source_anchor=str(cit.get("span_start", 0)),
                    ref=None,
                ))
    
    # Leading indicators
    if plan.leading_indicators:
        lines.append("- **مؤشرات القياس**:")
        for ind in plan.leading_indicators[:3]:
            source = ind.get("source", "غير منصوص")
            lines.append(f"  - {ind.get('indicator_ar', '')} ({source})")
    
    # Risks
    if plan.risk_of_imbalance:
        lines.append("- **تحذيرات**:")
        for risk in plan.risk_of_imbalance[:2]:
            lines.append(f"  - {risk.get('risk_ar', '')}")
    
    return "\n".join(lines), citations


def compose_global_synthesis_answer(
    *,
    plan: WorldModelPlan,
    question_ar: str,
    packets: list[dict[str, Any]],
) -> tuple[str, list[Citation], list[dict[str, Any]]]:
    """Compose a global synthesis answer using the world model plan.
    
    Args:
        plan: World model plan with loops, interventions, simulations
        question_ar: The original question
        packets: Evidence packets for additional citations
        
    Returns:
        Tuple of (answer_ar, citations, used_edges)
    """
    all_citations: list[Citation] = []
    sections: list[str] = []
    
    # Opening
    sections.append("## الرؤية الشاملة للإطار")
    
    # Check if we have enough material
    if not plan.loops and not plan.interventions:
        sections.append("")
        sections.append("### ما يمكن دعمه من الأدلة المسترجعة")
        sections.append("- لا توجد روابط آلية مؤصّلة كافية في edge_justification_span لبناء تحليل شامل.")
        sections.append("")
        sections.append("### ما لا يمكن الجزم به من الأدلة الحالية")
        sections.append("- الحلقات السببية بين الركائز")
        sections.append("- خطوات التدخل المبنية على الإطار")
        
        # Add some content from packets if available
        if packets:
            sections.append("")
            sections.append("### المعلومات المتاحة من الإطار")
            for p in packets[:3]:
                text = str(p.get("text_ar", ""))[:200]
                if text:
                    sections.append(f"- {text}")
                    if p.get("chunk_id"):
                        all_citations.append(Citation(
                            chunk_id=str(p["chunk_id"]),
                            source_anchor=str(p.get("source_anchor", "")),
                            ref=None,
                        ))
        
        return "\n".join(sections), all_citations, []
    
    # Pillar coverage summary
    all_pillars = ["P001", "P002", "P003", "P004", "P005"]
    pillar_names = {
        "P001": "الروحية",
        "P002": "العاطفية", 
        "P003": "الفكرية",
        "P004": "البدنية",
        "P005": "الاجتماعية",
    }
    
    covered_names = [pillar_names.get(p, p) for p in plan.covered_pillars if p in pillar_names]
    uncovered = [pillar_names.get(p, p) for p in all_pillars if p not in plan.covered_pillars]
    
    sections.append("")
    sections.append(f"**الركائز المغطاة**: {', '.join(covered_names) if covered_names else 'لا توجد'}")
    if uncovered:
        sections.append(f"**الركائز غير المغطاة في هذا التحليل**: {', '.join(uncovered)}")
    
    # Causal loops section
    if plan.loops:
        sections.append("")
        sections.append("### الحلقات السببية")
        
        for loop in plan.loops:
            loop_text, loop_cits = _format_loop_section(loop)
            sections.append(loop_text)
            all_citations.extend(loop_cits)
    
    # Intervention plans section
    if plan.interventions:
        sections.append("")
        sections.append("### خطة التدخل")
        
        for intervention in plan.interventions:
            int_text, int_cits = _format_intervention_section(intervention)
            sections.append(int_text)
            all_citations.extend(int_cits)
    
    # Simulation summary (if available)
    if plan.simulations:
        sections.append("")
        sections.append("### محاكاة التأثير")
        
        for sim in plan.simulations[:1]:
            sections.append(f"- {sim.label_ar}")
            if sim.propagation_steps:
                affected = len(sim.final_state)
                sections.append(f"- عدد العناصر المتأثرة: {affected}")
    
    # Executive summary
    sections.append("")
    sections.append("### خلاصة تنفيذية (3 نقاط)")
    
    summaries = []
    if plan.loops:
        loop_count = len(plan.loops)
        summaries.append(f"- يتضمن التحليل {loop_count} حلقة سببية مترابطة")
    
    if plan.interventions:
        for intv in plan.interventions[:1]:
            step_count = len(intv.steps)
            summaries.append(f"- خطة التدخل تتكون من {step_count} خطوات متسلسلة")
    
    summaries.append(f"- التغطية: {len(plan.covered_pillars)}/5 ركائز")
    
    for s in summaries[:3]:
        sections.append(s)
    
    # Add citations from packets for additional grounding
    seen_chunks: set[str] = {c.chunk_id for c in all_citations}
    for p in packets[:5]:
        chunk_id = str(p.get("chunk_id", ""))
        if chunk_id and chunk_id not in seen_chunks:
            all_citations.append(Citation(
                chunk_id=chunk_id,
                source_anchor=str(p.get("source_anchor", "")),
                ref=None,
            ))
            seen_chunks.add(chunk_id)
    
    return "\n".join(sections), all_citations, list(plan.used_edges)


def check_global_synthesis_requirements(
    plan: WorldModelPlan,
) -> tuple[bool, list[str]]:
    """Check if plan meets global synthesis requirements.
    
    Requirements:
    - At least 2 loops
    - At least 1 intervention plan
    - Coverage of all 5 pillars (soft requirement)
    
    Args:
        plan: The world model plan
        
    Returns:
        Tuple of (meets_requirements, list_of_issues)
    """
    issues: list[str] = []
    
    if len(plan.loops) < 2:
        issues.append(f"INSUFFICIENT_LOOPS: {len(plan.loops)}/2")
    
    if len(plan.interventions) < 1:
        issues.append("MISSING_INTERVENTION_PLAN")
    
    if len(plan.covered_pillars) < 5:
        # This is a soft requirement - don't fail, just note it
        issues.append(f"INCOMPLETE_PILLAR_COVERAGE: {len(plan.covered_pillars)}/5")
    
    # Check if we have any grounded evidence
    has_evidence = False
    for loop in plan.loops:
        if loop.evidence_spans:
            has_evidence = True
            break
    
    if not has_evidence:
        issues.append("NO_GROUNDED_EVIDENCE")
    
    # Hard requirements: loops and interventions
    meets = len(plan.loops) >= 2 and len(plan.interventions) >= 1
    
    return meets, issues
