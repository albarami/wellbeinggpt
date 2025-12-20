#!/usr/bin/env python3
"""
Arabic Stress Test Script (12 Questions)

Runs 12 Arabic stress-test questions against the live backend API,
evaluates responses using the stakeholder checklist, and generates
a readable UTF-8 markdown report.

Usage:
    python scripts/stress_test_arabic_12.py

Output:
    eval/reports/stress_test_12_report.md
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

# Ensure UTF-8 output on Windows
os.environ["PYTHONIOENCODING"] = "utf-8"
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Load .env file
env_file = project_root / ".env"
if env_file.exists():
    with open(env_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                key, _, value = line.partition("=")
                key = key.strip()
                value = value.strip()
                if key:
                    os.environ[key] = value

from httpx import AsyncClient, ASGITransport

# =============================================================================
# 12 Arabic Stress-Test Questions
# =============================================================================

STRESS_QUESTIONS: list[dict[str, Any]] = [
    # A) Global synthesis (tests "super-smart" integration across all pillars)
    {
        "id": "stress-A1",
        "category": "global_synthesis",
        "question_ar": (
            'كيف تُسهم "الحياة الطيبة" في ازدهار الإنسان والمجتمع وفق الإطار؟ '
            "ابنِ حجة متكاملة تربط الركائز الخمس مع أمثلة قصيرة، "
            "واذكر حدود ما يمكن الجزم به من نص الإطار."
        ),
        "mode": "answer",
        "checks": ["coverage", "citations", "boundaries"],
        "require_edges": False,
        "require_boundaries": True,
    },
    {
        "id": "stress-A2",
        "category": "global_synthesis",
        "question_ar": (
            'ما الذي يميّز هذا الإطار عن "التنمية المادية" وحدها؟ '
            "قدّم مقارنة مفهومية من داخل الإطار فقط، مع أدلة."
        ),
        "mode": "answer",
        "checks": ["coverage", "citations"],
        "require_edges": False,
        "require_boundaries": False,
    },
    # B) Cross-pillar path (tests used_edges + argument_chains + per-edge evidence)
    {
        "id": "stress-B1",
        "category": "cross_pillar_path",
        "question_ar": (
            'اشرح مسارًا عبر الركائز يبدأ من "الحياة البدنية" وينتهي إلى "الحياة الروحية": '
            "اذكر كل خطوة (قيمة→قيمة) ونوع العلاقة (تمكين/تعزيز/تكامل/شرط) مع دليل لكل رابط."
        ),
        "mode": "answer",
        "checks": ["coverage", "used_edges", "argument_chains", "citations"],
        "require_edges": True,
        "require_boundaries": False,
    },
    {
        "id": "stress-B2",
        "category": "cross_pillar_path",
        "question_ar": (
            'ابنِ "حلقة تكامل" من 4 خطوات تربط: بدني → فكري → عاطفي → اجتماعي '
            "(أو أي ترتيب تدعمه الأدلة) مع تبرير كل خطوة وحدود الربط إن وُجدت."
        ),
        "mode": "answer",
        "checks": ["coverage", "used_edges", "argument_chains", "citations", "boundaries"],
        "require_edges": True,
        "require_boundaries": True,
    },
    # C) Network build (tests ≥3 pillars + non-trivial synthesis)
    {
        "id": "stress-C1",
        "category": "network_build",
        "question_ar": (
            "اختر قيمة محورية واحدة داخل الإطار ثم ابنِ شبكة تربطها بثلاث ركائز أخرى على الأقل. "
            'لكل رابط: نوع العلاقة + دليل + "وجه الدلالة".'
        ),
        "mode": "answer",
        "checks": ["coverage", "used_edges", "argument_chains", "citations"],
        "require_edges": True,
        "require_boundaries": False,
    },
    {
        "id": "stress-C2",
        "category": "network_build",
        "question_ar": (
            "ما أقوى رابطين عبر الركائز في الإطار؟ (حسب الأدلة الموجودة) ولماذا؟ "
            "قدّم تفسيرًا موجزًا مع الاستشهاد."
        ),
        "mode": "answer",
        "checks": ["coverage", "used_edges", "argument_chains", "citations"],
        "require_edges": True,
        "require_boundaries": False,
    },
    # D) Compare / differentiate (tests concept completeness + matrix)
    {
        "id": "stress-D1",
        "category": "compare_differentiate",
        "question_ar": (
            'فرّق بين "الإيمان" و"العبادة" و"التزكية" داخل الإطار من حيث: '
            "التعريف، المظهر العملي، والخطأ الشائع في الخلط بينها. "
            "(أريد جدولًا صغيرًا + أدلة)."
        ),
        "mode": "answer",
        "checks": ["coverage", "citations", "style"],
        "require_edges": False,
        "require_boundaries": False,
    },
    {
        "id": "stress-D2",
        "category": "compare_differentiate",
        "question_ar": (
            'ما الفرق بين "التوازن" وبين مفهوم قريب منه داخل الإطار؟ '
            "عرّف كل واحد بدليل، ثم أعط مثالًا تطبيقيًا لكل منهما."
        ),
        "mode": "answer",
        "checks": ["coverage", "citations"],
        "require_edges": False,
        "require_boundaries": False,
    },
    # E) Boundaries / limits (tests scholar "ضوابط" + honesty)
    {
        "id": "stress-E1",
        "category": "boundaries_limits",
        "question_ar": (
            'اذكر "حدود" الاستدلال في الإطار: ما الذي ينص عليه الإطار بوضوح؟ '
            "وما الذي لا نستطيع الجزم به لأنه غير منصوص عليه؟"
        ),
        "mode": "answer",
        "checks": ["coverage", "citations", "boundaries"],
        "require_edges": False,
        "require_boundaries": True,
    },
    {
        "id": "stress-E2",
        "category": "boundaries_limits",
        "question_ar": (
            'متى يتحول مفهوم "الاعتدال/التوازن" إلى إفراط أو تفريط داخل الإطار؟ '
            "إن لم يذكر الإطار ضوابط صريحة، صرّح بذلك وبيّن ما يمكن دعمه فقط."
        ),
        "mode": "answer",
        "checks": ["coverage", "citations", "boundaries"],
        "require_edges": False,
        "require_boundaries": True,
    },
    # F) Natural conversation (tests "soul" + coherence without bullet spam)
    {
        "id": "stress-F1",
        "category": "natural_conversation",
        "question_ar": (
            "تحدث معي بأسلوب حواري طبيعي: إذا أردت أن أعيش \"حياة طيبة\" خلال أسبوع واحد، "
            "ما الفكرة المركزية التي يجب أن أبدأ بها وفق هذا الإطار؟ "
            "اربطها بالركائز الخمس بشكل طبيعي مع أدلة قليلة مركزة."
        ),
        "mode": "natural_chat",
        "checks": ["coverage", "citations", "style"],
        "require_edges": False,
        "require_boundaries": False,
    },
    {
        "id": "stress-F2",
        "category": "natural_conversation",
        "question_ar": (
            "سؤال فلسفي عملي: لماذا لا يفصل الإطار بين الجسد والروح؟ "
            "وكيف ينعكس ذلك على طريقة عيش الإنسان يوميًا؟ "
            "(استشهد من الإطار فقط)."
        ),
        "mode": "natural_chat",
        "checks": ["coverage", "citations", "style"],
        "require_edges": False,
        "require_boundaries": False,
    },
]


# =============================================================================
# Scoring Functions
# =============================================================================


def check_coverage(answer: str, question: dict[str, Any]) -> tuple[int, str]:
    """
    Check if answer covers the question (0-2 score).
    
    - 2: Substantive answer addressing the question
    - 1: Partial coverage or some evasion
    - 0: Refuses or completely evades
    """
    if not answer or len(answer.strip()) < 50:
        return 0, "Empty or too short answer"
    
    # Check for refusal patterns
    refusal_markers = [
        "لا يوجد في البيانات",
        "لا تتوفر معلومات",
        "لا أستطيع",
        "خارج نطاق",
    ]
    for marker in refusal_markers:
        if marker in answer:
            return 0, f"Refusal detected: '{marker}'"
    
    # Check for substantive content
    category = question.get("category", "")
    
    if category == "global_synthesis":
        # Should mention multiple pillars - check various forms
        pillar_markers = ["الروحية", "البدنية", "الفكرية", "العاطفية", "الاجتماعية"]
        pillar_alt = ["روحي", "بدني", "فكري", "عاطفي", "اجتماعي", "P001", "P002", "P003", "P004", "P005"]
        found = sum(1 for m in pillar_markers if m in answer)
        found_alt = sum(1 for m in pillar_alt if m in answer)
        total = min(5, found + found_alt // 2)  # Avoid double counting
        if total >= 4:
            return 2, f"Covers {total}/5 pillars"
        elif total >= 2:
            return 1, f"Partial coverage: {total}/5 pillars"
        else:
            return 0, f"Insufficient pillar coverage: {total}/5"
    
    elif category in ["cross_pillar_path", "network_build"]:
        # Should have path/link language
        path_markers = ["→", "تمكين", "تعزيز", "تكامل", "شرط", "يؤدي", "يربط", "العلاقة", "ENABLES", "REINFORCES", "COMPLEMENTS"]
        found = sum(1 for m in path_markers if m in answer)
        if found >= 3:
            return 2, f"Good path/link structure ({found} markers)"
        elif found >= 1:
            return 1, f"Some path structure ({found} markers)"
        else:
            return 0, "No path/link structure found"
    
    elif category == "compare_differentiate":
        # Should have comparison structure - be generous, detect various formats
        compare_markers = [
            "الفرق", "مقارنة", "يختلف", "بينما", "أما", "التعريف", "المظهر",
            # Scholar format markers
            "تعريف المفهوم", "التأصيل", "وجه الدلالة",
            # Concept-specific markers
            "التوازن", "الاعتدال", "الإيمان", "العبادة", "التزكية",
        ]
        found = sum(1 for m in compare_markers if m in answer)
        # Check for definition structure (تعريف for each concept)
        def_count = answer.count("تعريف") + answer.count("عرّف") + answer.count("وجه الدلالة")
        if found >= 3 or def_count >= 2:
            return 2, f"Good comparison structure ({found} markers, {def_count} definitions)"
        elif found >= 1 or def_count >= 1:
            return 1, f"Some comparison ({found} markers, {def_count} definitions)"
        else:
            return 0, "No comparison structure"
    
    elif category == "boundaries_limits":
        # Should discuss limits - be comprehensive
        limit_markers = [
            "حدود", "ضوابط", "لا يمكن", "غير منصوص", "بوضوح", "الجزم",
            # Scholar format markers
            "تنبيهات", "أخطاء شائعة", "خلاصة",
            # Boundary-specific
            "إفراط", "تفريط", "اعتدال", "توازن", "يجب", "ينبغي",
        ]
        found = sum(1 for m in limit_markers if m in answer)
        # Check for warnings section
        has_warnings = "تنبيهات" in answer or "أخطاء شائعة" in answer
        if found >= 3 or (found >= 2 and has_warnings):
            return 2, f"Discusses boundaries ({found} markers)"
        elif found >= 1:
            return 1, f"Partial boundary discussion ({found} markers)"
        else:
            return 0, "No boundary discussion"
    
    elif category == "natural_conversation":
        # Should be substantive and flowing
        if len(answer) >= 300:
            return 2, "Substantive natural response"
        elif len(answer) >= 150:
            return 1, "Moderate natural response"
        else:
            return 0, "Too brief for natural conversation"
    
    # Default: check length
    if len(answer) >= 400:
        return 2, "Substantive answer"
    elif len(answer) >= 150:
        return 1, "Moderate answer"
    else:
        return 0, "Brief answer"


def check_depth(response: dict[str, Any], question: dict[str, Any]) -> tuple[int, str]:
    """
    Check answer depth quality (0-2 score).
    
    Measures real depth beyond length:
    - Scholar structure sections
    - Grounded edges with justification
    - Evidence quality
    """
    answer = response.get("answer_ar") or ""
    
    # Check for scholar structure sections (depth indicators)
    depth_sections = [
        "تعريف المفهوم داخل الإطار",
        "التأصيل والأدلة",
        "الربط بين الركائز",
        "تنبيهات وأخطاء شائعة",
        "خلاصة تنفيذية",
        "تطبيق عملي",
    ]
    sections_found = sum(1 for s in depth_sections if s in answer)
    
    # Check for grounded citations (not just existence but quality)
    citations = response.get("citations") or []
    citations_with_quotes = sum(1 for c in citations if len(str(c.get("quote") or "")) > 20)
    
    # Check for edge-grounded reasoning
    graph_trace = response.get("graph_trace") or {}
    used_edges = graph_trace.get("used_edges") or []
    edges_with_spans = sum(1 for e in used_edges if len(e.get("justification_spans") or []) > 0)
    
    # Check for per-step evidence (خطوة 1, خطوة 2, etc.)
    step_count = answer.count("خطوة")
    
    # Check for evidence markers
    evidence_markers = ["شاهد:", "دليل:", "قال تعالى", "قوله تعالى", "حديث", "رواه"]
    evidence_found = sum(1 for m in evidence_markers if m in answer)
    
    # Score calculation
    score = 0
    reasons = []
    
    if sections_found >= 3:
        score += 1
        reasons.append(f"{sections_found} sections")
    
    if citations_with_quotes >= 2 or edges_with_spans >= 2 or evidence_found >= 2:
        score += 1
        reasons.append(f"{citations_with_quotes} quoted cites, {edges_with_spans} grounded edges, {evidence_found} evidence markers")
    
    if score == 0 and (sections_found >= 1 or citations_with_quotes >= 1 or step_count >= 2):
        score = 1
        reasons.append(f"Partial depth: {sections_found} sections, {step_count} steps")
    
    if not reasons:
        reasons.append("Shallow answer structure")
    
    return score, "; ".join(reasons)


def check_used_edges(response: dict[str, Any], question: dict[str, Any]) -> tuple[int, str]:
    """
    Check for used_edges in graph trace (0-2 score).
    
    Only scored for questions 3-6 (cross_pillar_path, network_build).
    """
    if not question.get("require_edges"):
        return 2, "N/A (not required for this question)"
    
    # Try to find used_edges in response
    graph_trace = response.get("graph_trace") or {}
    used_edges = graph_trace.get("used_edges") or []
    
    # Also check in debug info
    debug = response.get("debug") or {}
    
    if len(used_edges) >= 3:
        return 2, f"Has {len(used_edges)} used_edges"
    elif len(used_edges) >= 1:
        return 1, f"Has {len(used_edges)} used_edge(s) (want ≥3)"
    else:
        return 0, "No used_edges found"


def check_argument_chains(response: dict[str, Any], question: dict[str, Any]) -> tuple[int, str]:
    """
    Check for argument_chains in graph trace (0-2 score).
    
    Only scored for questions 3-6 (cross_pillar_path, network_build).
    """
    if not question.get("require_edges"):
        return 2, "N/A (not required for this question)"
    
    graph_trace = response.get("graph_trace") or {}
    argument_chains = graph_trace.get("argument_chains") or []
    
    if len(argument_chains) >= 2:
        return 2, f"Has {len(argument_chains)} argument_chains"
    elif len(argument_chains) >= 1:
        return 1, f"Has {len(argument_chains)} argument_chain(s)"
    else:
        return 0, "No argument_chains found"


def check_citations(response: dict[str, Any], answer: str) -> tuple[int, str]:
    """
    Check citation quality (0-2 score).
    
    - Few but strong (no excessive repetition)
    - Actually grounded quotes
    """
    citations = response.get("citations") or []
    
    if not citations:
        return 0, "No citations"
    
    # Check for uniqueness
    chunk_ids = [c.get("chunk_id") for c in citations]
    unique_chunks = len(set(chunk_ids))
    
    # Check for meaningful quotes
    quotes_with_content = sum(1 for c in citations if len(str(c.get("quote") or c.get("ref") or "")) > 10)
    
    if unique_chunks >= 2 and quotes_with_content >= 2:
        return 2, f"{unique_chunks} unique citations with content"
    elif unique_chunks >= 1 or quotes_with_content >= 1:
        return 1, f"{unique_chunks} unique, {quotes_with_content} with quotes"
    else:
        return 0, "Citations lack uniqueness or content"


def check_boundaries(answer: str, question: dict[str, Any]) -> tuple[int, str]:
    """
    Check if answer mentions limits/boundaries when needed (0-2 score).
    """
    if not question.get("require_boundaries"):
        return 2, "N/A (boundaries not required)"
    
    boundary_markers = [
        "غير منصوص",
        "لا يمكن الجزم",
        "حدود",
        "ضوابط",
        "لم يُنص",
        "لم يذكر الإطار",
        "ما لا يمكن",
        "لا نستطيع الجزم",
    ]
    
    found = [m for m in boundary_markers if m in answer]
    
    if len(found) >= 2:
        return 2, f"Clear boundary markers: {found[:2]}"
    elif len(found) >= 1:
        return 1, f"Some boundary mention: {found[0]}"
    else:
        return 0, "No boundary acknowledgment"


def check_style(answer: str, question: dict[str, Any]) -> tuple[int, str]:
    """
    Check answer style (0-2 score).
    
    - natural_chat mode: flowing prose, no bullet spam
    - answer mode: structured, can have bullets
    """
    mode = question.get("mode", "answer")
    
    if mode == "natural_chat":
        # Count bullet markers
        bullet_count = answer.count("\n-") + answer.count("\n•") + answer.count("\n*")
        newline_count = answer.count("\n")
        
        # Natural should have fewer bullets relative to content
        if newline_count > 0:
            bullet_ratio = bullet_count / newline_count
        else:
            bullet_ratio = 0
        
        if bullet_ratio < 0.3 and len(answer) >= 200:
            return 2, f"Natural flowing style (bullet ratio: {bullet_ratio:.2f})"
        elif bullet_ratio < 0.5:
            return 1, f"Somewhat natural (bullet ratio: {bullet_ratio:.2f})"
        else:
            return 0, f"Too many bullets for natural style (ratio: {bullet_ratio:.2f})"
    
    else:
        # Structured answer mode - should be organized
        structure_markers = [
            "تعريف المفهوم",
            "التأصيل",
            "الربط بين الركائز",
            "تنبيهات",
            "خلاصة",
            "أولاً",
            "ثانياً",
        ]
        found = sum(1 for m in structure_markers if m in answer)
        
        if found >= 3 or ("\n-" in answer and len(answer) >= 300):
            return 2, f"Well-structured ({found} section markers)"
        elif found >= 1 or len(answer) >= 200:
            return 1, f"Some structure ({found} markers)"
        else:
            return 0, "Lacks structure"


def score_response(response: dict[str, Any], question: dict[str, Any]) -> dict[str, Any]:
    """
    Score a single response against all applicable checks.
    
    Returns dict with individual scores and total.
    """
    answer = response.get("answer_ar") or ""
    checks = question.get("checks", [])
    
    scores: dict[str, tuple[int, str]] = {}
    
    # Always check coverage
    scores["coverage"] = check_coverage(answer, question)
    
    # Always check depth (new!)
    scores["depth"] = check_depth(response, question)
    
    # Check used_edges for cross-pillar questions
    if "used_edges" in checks or question.get("require_edges"):
        scores["used_edges"] = check_used_edges(response, question)
    
    # Check argument_chains for cross-pillar questions
    if "argument_chains" in checks or question.get("require_edges"):
        scores["argument_chains"] = check_argument_chains(response, question)
    
    # Always check citations
    scores["citations"] = check_citations(response, answer)
    
    # Check boundaries when required
    if "boundaries" in checks or question.get("require_boundaries"):
        scores["boundaries"] = check_boundaries(answer, question)
    
    # Check style
    if "style" in checks or question.get("mode") == "natural_chat":
        scores["style"] = check_style(answer, question)
    
    # Calculate totals
    total_score = sum(s[0] for s in scores.values())
    max_score = len(scores) * 2
    
    return {
        "scores": scores,
        "total": total_score,
        "max": max_score,
        "pass": total_score >= (max_score * 0.6),  # 60% threshold
    }


# =============================================================================
# Report Generation
# =============================================================================


def generate_report(results: list[dict[str, Any]], run_time: float) -> str:
    """Generate markdown report from results."""
    lines: list[str] = []
    
    # Header
    lines.append("# Arabic Stress Test Report (12 Questions)")
    lines.append("")
    lines.append(f"**Run date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"**Run time**: {run_time:.1f} seconds")
    lines.append("")
    
    # Calculate totals
    total_score = sum(r["scoring"]["total"] for r in results)
    total_max = sum(r["scoring"]["max"] for r in results)
    pass_count = sum(1 for r in results if r["scoring"]["pass"])
    
    lines.append(f"**Total score**: {total_score}/{total_max} ({100*total_score/total_max:.1f}%)")
    lines.append(f"**Pass rate**: {pass_count}/12 ({100*pass_count/12:.1f}%)")
    lines.append("")
    
    # Summary table
    lines.append("## Summary Table")
    lines.append("")
    lines.append("| # | ID | Category | Score | Pass |")
    lines.append("|---|-----|----------|-------|------|")
    
    for i, r in enumerate(results, 1):
        q = r["question"]
        s = r["scoring"]
        pass_mark = "YES" if s["pass"] else "NO"
        lines.append(f"| {i} | {q['id']} | {q['category']} | {s['total']}/{s['max']} | {pass_mark} |")
    
    lines.append("")
    
    # Detailed results
    lines.append("## Detailed Results")
    lines.append("")
    
    for i, r in enumerate(results, 1):
        q = r["question"]
        resp = r["response"]
        s = r["scoring"]
        
        lines.append(f"### {i}) {q['id']} ({q['category']})")
        lines.append("")
        lines.append(f"**Mode**: `{q['mode']}`")
        lines.append(f"**Score**: {s['total']}/{s['max']} {'PASS' if s['pass'] else 'FAIL'}")
        lines.append(f"**Latency**: {r.get('latency_ms', 0)} ms")
        lines.append(f"**Contract**: `{resp.get('contract_outcome', 'N/A')}`")
        lines.append("")
        
        # Individual scores
        lines.append("**Checklist Scores**:")
        lines.append("")
        for check_name, (score, reason) in s["scores"].items():
            lines.append(f"- {check_name}: {score}/2 — {reason}")
        lines.append("")
        
        # Question
        lines.append("**Question**:")
        lines.append("")
        lines.append(f"> {q['question_ar']}")
        lines.append("")
        
        # Answer (truncated)
        answer = resp.get("answer_ar") or "(no answer)"
        lines.append("**Answer** (first 1500 chars):")
        lines.append("")
        lines.append("```")
        lines.append(answer[:1500])
        if len(answer) > 1500:
            lines.append("... [truncated]")
        lines.append("```")
        lines.append("")
        
        # Citations
        citations = resp.get("citations") or []
        lines.append(f"**Citations** ({len(citations)} total):")
        lines.append("")
        if not citations:
            lines.append("- (none)")
        else:
            for c in citations[:8]:
                chunk_id = c.get("chunk_id") or c.get("source_id") or ""
                quote = c.get("quote") or c.get("ref") or ""
                quote_short = quote[:80] + "..." if len(quote) > 80 else quote
                lines.append(f"- `{chunk_id}`: {quote_short}")
        lines.append("")
        
        # Graph trace summary (for cross-pillar questions)
        if q.get("require_edges"):
            graph_trace = resp.get("graph_trace") or {}
            used_edges = graph_trace.get("used_edges") or []
            argument_chains = graph_trace.get("argument_chains") or []
            
            lines.append("**Graph Trace**:")
            lines.append("")
            lines.append(f"- used_edges: {len(used_edges)}")
            lines.append(f"- argument_chains: {len(argument_chains)}")
            
            if used_edges:
                lines.append("")
                lines.append("```json")
                lines.append(json.dumps(used_edges[:3], ensure_ascii=False, indent=2))
                if len(used_edges) > 3:
                    lines.append(f"... and {len(used_edges) - 3} more")
                lines.append("```")
            lines.append("")
        
        lines.append("---")
        lines.append("")
    
    return "\n".join(lines)


# =============================================================================
# Main Runner
# =============================================================================


async def run_single_question(
    client: AsyncClient,
    question: dict[str, Any],
) -> dict[str, Any]:
    """Run a single question and return response with timing.
    
    Uses /ask/ui endpoint to get full graph_trace with used_edges and argument_chains.
    """
    t0 = time.perf_counter()
    
    try:
        # Use /ask/ui to get full response including graph_trace
        resp = await client.post(
            "/ask/ui",
            json={
                "question": question["question_ar"],
                "mode": question.get("mode", "answer"),
                "language": "ar",
            },
            timeout=180.0,  # 3 minutes timeout for complex questions
        )
        
        if resp.status_code != 200:
            return {
                "error": f"HTTP {resp.status_code}: {resp.text[:200]}",
                "answer_ar": "",
                "citations": [],
                "graph_trace": {},
            }
        
        data = resp.json()
        latency_ms = int((time.perf_counter() - t0) * 1000)
        
        # Extract from /ask/ui response structure
        final = data.get("final") or {}
        graph_trace = data.get("graph_trace") or {}
        citations_spans = data.get("citations_spans") or []
        
        # Build normalized response for scoring
        return {
            "answer_ar": final.get("answer_ar") or "",
            "citations": [
                {
                    "chunk_id": c.get("chunk_id") or "",
                    "source_id": c.get("source_id") or "",
                    "quote": c.get("quote") or "",
                    "source_anchor": c.get("source_anchor") or "",
                }
                for c in citations_spans
            ],
            "graph_trace": {
                "used_edges": graph_trace.get("used_edges") or [],
                "argument_chains": graph_trace.get("argument_chains") or [],
            },
            "not_found": final.get("not_found", False),
            "confidence": final.get("confidence", "low"),
            "contract_outcome": data.get("contract_outcome") or "",
            "contract_reasons": data.get("contract_reasons") or [],
            "latency_ms": data.get("latency_ms") or latency_ms,
        }
    
    except Exception as e:
        return {
            "error": str(e),
            "answer_ar": "",
            "citations": [],
            "graph_trace": {},
            "latency_ms": int((time.perf_counter() - t0) * 1000),
        }


async def main() -> None:
    """Main entry point."""
    print("=" * 60)
    print("Arabic Stress Test (12 Questions)")
    print("=" * 60)
    print()
    
    # Import app here to avoid import errors before env is loaded
    from apps.api.main import app
    
    results: list[dict[str, Any]] = []
    total_start = time.perf_counter()
    
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        for i, question in enumerate(STRESS_QUESTIONS, 1):
            print(f"[{i}/12] Running: {question['id']} ({question['category']})")
            print(f"        Question: {question['question_ar'][:60]}...")
            
            response = await run_single_question(client, question)
            
            if "error" in response and response["error"]:
                print(f"        ERROR: {response['error'][:100]}")
            else:
                print(f"        Answer length: {len(response.get('answer_ar') or '')} chars")
                print(f"        Citations: {len(response.get('citations') or [])}")
                graph_trace = response.get("graph_trace") or {}
                used_edges = graph_trace.get("used_edges") or []
                argument_chains = graph_trace.get("argument_chains") or []
                print(f"        Graph: {len(used_edges)} edges, {len(argument_chains)} chains")
                print(f"        Contract: {response.get('contract_outcome', 'N/A')}")
                print(f"        Latency: {response.get('latency_ms', 0)} ms")
            
            scoring = score_response(response, question)
            print(f"        Score: {scoring['total']}/{scoring['max']} {'PASS' if scoring['pass'] else 'FAIL'}")
            print()
            
            results.append({
                "question": question,
                "response": response,
                "scoring": scoring,
                "latency_ms": response.get("latency_ms", 0),
            })
    
    total_time = time.perf_counter() - total_start
    
    # Generate report
    report = generate_report(results, total_time)
    
    # Write report
    report_path = project_root / "eval" / "reports" / "stress_test_12_report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")
    
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    total_score = sum(r["scoring"]["total"] for r in results)
    total_max = sum(r["scoring"]["max"] for r in results)
    pass_count = sum(1 for r in results if r["scoring"]["pass"])
    
    print(f"Total score: {total_score}/{total_max} ({100*total_score/total_max:.1f}%)")
    print(f"Pass rate: {pass_count}/12 ({100*pass_count/12:.1f}%)")
    print(f"Total time: {total_time:.1f}s")
    print()
    print(f"Report written to: {report_path}")


if __name__ == "__main__":
    asyncio.run(main())
