"""
Muḥāsibī A/B Test Harness

Runs a fixed set of wellbeing guidance questions in both baseline and Muḥāsibī modes,
scores them using a rubric, and produces a comparison table.

Usage:
    python scripts/muhasibi_ab_test.py

Env vars required:
    DATABASE_URL - Postgres connection string
    AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY - For LLM
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx

from dotenv import load_dotenv

from apps.api.main import app


def _require_env(name: str) -> str:
    val = (os.getenv(name) or "").strip()
    if not val:
        raise RuntimeError(f"Missing required env var: {name}")
    return val


# Test questions for A/B testing (from plan section E2)
AB_TEST_QUESTIONS = [
    {"id": "ab-01", "question_ar": "أنا أغضب بسرعة. كيف أطبق ضبط الانفعالات مع دليل؟"},
    {"id": "ab-02", "question_ar": "أشعر بقلق شديد. كيف أحقق السكينة؟"},
    {"id": "ab-03", "question_ar": "كيف أطبق التقبل دون استسلام سلبي؟"},
    {"id": "ab-04", "question_ar": "كيف أجمع بين الطموح والاعتدال؟"},
    {"id": "ab-05", "question_ar": "كيف أربط بين النية الصادقة والصمود؟"},
    {"id": "ab-06", "question_ar": "كيف أوازن بين العمل والراحة الكافية؟"},
    {"id": "ab-07", "question_ar": "كيف أمارس التقويم الذاتي بطريقة شرعية وعملية؟"},
    {"id": "ab-08", "question_ar": "كيف أحول تجربة فشل إلى نضج؟"},
    {"id": "ab-09", "question_ar": "كيف أتعامل مع خصومة اجتماعية عبر الصفح والعدالة؟"},
    {"id": "ab-10", "question_ar": "كيف أستخدم التفكير النقدي والتبين لتجنب الشائعات؟"},
]


@dataclass
class RubricScore:
    """Rubric scoring for a response."""
    evidence_grounding: int  # 0-5: Citations accurate and relevant
    refusal_correctness: int  # 0-5: Fewer false answers where evidence is weak
    depth_structure: int  # 0-5: Clear steps (baseline) or LISTEN/PURPOSE/PATH/ACCOUNT/REFLECT (Muhasibi)
    cross_pillar_linkage: int  # 0-5: Cites >1 pillar when applicable
    actionability: int  # 0-5: Clear next steps without inventing claims
    
    @property
    def total(self) -> int:
        return (
            self.evidence_grounding +
            self.refusal_correctness +
            self.depth_structure +
            self.cross_pillar_linkage +
            self.actionability
        )
    
    @property
    def average(self) -> float:
        return self.total / 5.0


def score_response(response: dict[str, Any], mode: str) -> RubricScore:
    """
    Score a response using the rubric.
    
    Args:
        response: The API response.
        mode: "baseline" or "muhasibi".
        
    Returns:
        RubricScore with 0-5 scores.
    """
    # Evidence grounding: based on citations
    citations = response.get("citations", [])
    if response.get("not_found"):
        evidence_score = 3  # Correct refusal gets moderate score
    elif len(citations) >= 5:
        evidence_score = 5
    elif len(citations) >= 3:
        evidence_score = 4
    elif len(citations) >= 1:
        evidence_score = 3
    else:
        evidence_score = 1
    
    # Refusal correctness: based on not_found + citations consistency
    if response.get("not_found") and len(citations) == 0:
        refusal_score = 5  # Correct refusal
    elif not response.get("not_found") and len(citations) > 0:
        refusal_score = 5  # Correct answer with citations
    elif response.get("not_found") and len(citations) > 0:
        refusal_score = 2  # Inconsistent
    else:
        refusal_score = 3  # Uncertain
    
    # Depth structure: based on response structure
    answer = response.get("answer_ar", "")
    path_plan = response.get("path_plan_ar", [])
    listen_summary = response.get("listen_summary_ar", "")
    
    if mode == "muhasibi":
        # Check for Muhasibi markers
        structure_markers = 0
        if listen_summary and len(listen_summary) > 10:
            structure_markers += 1
        if len(path_plan) >= 3:
            structure_markers += 1
        if "تأمل" in answer or "تدبر" in answer or "انعكاس" in answer:
            structure_markers += 1
        if len(answer) > 200:
            structure_markers += 1
        if response.get("purpose", {}).get("constraints_ar"):
            structure_markers += 1
        
        depth_score = min(5, structure_markers + 1)
    else:
        # Baseline: simpler structure check
        if len(answer) > 100:
            depth_score = 3
        elif len(answer) > 50:
            depth_score = 2
        else:
            depth_score = 1
    
    # Cross-pillar linkage: check entities from different pillars
    entities = response.get("entities", [])
    # This is a heuristic - in reality we'd check pillar_id
    entity_types = set(e.get("type") for e in entities)
    if len(entities) >= 5:
        cross_pillar_score = 4
    elif len(entities) >= 3:
        cross_pillar_score = 3
    elif len(entities) >= 1:
        cross_pillar_score = 2
    else:
        cross_pillar_score = 1
    
    # Actionability: check for action words
    action_patterns = [
        r"يمكنك",
        r"عليك",
        r"ينبغي",
        r"حاول",
        r"ابدأ",
        r"تجنب",
        r"احرص",
        r"خطوات",
    ]
    action_count = sum(1 for p in action_patterns if re.search(p, answer))
    if action_count >= 3:
        actionability_score = 5
    elif action_count >= 2:
        actionability_score = 4
    elif action_count >= 1:
        actionability_score = 3
    else:
        actionability_score = 2
    
    return RubricScore(
        evidence_grounding=evidence_score,
        refusal_correctness=refusal_score,
        depth_structure=depth_score,
        cross_pillar_linkage=cross_pillar_score,
        actionability=actionability_score,
    )


async def ask_question(client: httpx.AsyncClient, question: str, engine: str) -> dict[str, Any]:
    """
    Ask a question via the API.
    
    Args:
        client: HTTP client.
        question: The question in Arabic.
        engine: "baseline" or "muhasibi".
        
    Returns:
        The API response.
    """
    try:
        response = await client.post(
            "/ask",
            json={"question": question, "engine": engine},
            timeout=120.0,
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        return {"error": str(e), "not_found": True, "citations": [], "answer_ar": ""}


async def run_ab_test() -> dict[str, Any]:
    """
    Run the A/B test.
    
    Returns:
        Results dictionary.
    """
    results = {
        "generated_at": datetime.utcnow().isoformat(),
        "questions": [],
        "baseline_scores": [],
        "muhasibi_scores": [],
        "summary": {},
    }
    
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        for q in AB_TEST_QUESTIONS:
            print(f"\nTesting: {q['id']} - {q['question_ar'][:40]}...")
            
            question_result = {
                "id": q["id"],
                "question_ar": q["question_ar"],
                "baseline": {},
                "muhasibi": {},
            }
            
            # Run baseline
            print("  Running baseline...")
            baseline_response = await ask_question(client, q["question_ar"], "baseline")
            baseline_score = score_response(baseline_response, "baseline")
            question_result["baseline"] = {
                "response": baseline_response,
                "score": {
                    "evidence_grounding": baseline_score.evidence_grounding,
                    "refusal_correctness": baseline_score.refusal_correctness,
                    "depth_structure": baseline_score.depth_structure,
                    "cross_pillar_linkage": baseline_score.cross_pillar_linkage,
                    "actionability": baseline_score.actionability,
                    "total": baseline_score.total,
                    "average": baseline_score.average,
                },
            }
            results["baseline_scores"].append(baseline_score.average)
            
            # Run Muhasibi
            print("  Running Muhasibi...")
            muhasibi_response = await ask_question(client, q["question_ar"], "muhasibi")
            muhasibi_score = score_response(muhasibi_response, "muhasibi")
            question_result["muhasibi"] = {
                "response": muhasibi_response,
                "score": {
                    "evidence_grounding": muhasibi_score.evidence_grounding,
                    "refusal_correctness": muhasibi_score.refusal_correctness,
                    "depth_structure": muhasibi_score.depth_structure,
                    "cross_pillar_linkage": muhasibi_score.cross_pillar_linkage,
                    "actionability": muhasibi_score.actionability,
                    "total": muhasibi_score.total,
                    "average": muhasibi_score.average,
                },
            }
            results["muhasibi_scores"].append(muhasibi_score.average)
            
            # Compare
            diff = muhasibi_score.average - baseline_score.average
            question_result["diff"] = diff
            print(f"  Baseline: {baseline_score.average:.2f}, Muhasibi: {muhasibi_score.average:.2f}, Diff: {diff:+.2f}")
            
            results["questions"].append(question_result)
    
    # Calculate summary
    if results["baseline_scores"] and results["muhasibi_scores"]:
        baseline_avg = sum(results["baseline_scores"]) / len(results["baseline_scores"])
        muhasibi_avg = sum(results["muhasibi_scores"]) / len(results["muhasibi_scores"])
        overall_diff = muhasibi_avg - baseline_avg
        overall_diff_rounded = round(overall_diff, 2)
        
        results["summary"] = {
            "baseline_average": round(baseline_avg, 2),
            "muhasibi_average": round(muhasibi_avg, 2),
            "overall_difference": overall_diff_rounded,
            # Use the rounded metric for the pass/fail gate to avoid float jitter.
            "muhasibi_adds_value": overall_diff_rounded >= 1.0,  # Per plan: +1.0 threshold
            "questions_tested": len(results["questions"]),
        }
    
    return results


def print_results_table(results: dict[str, Any]) -> None:
    """Print a formatted results table."""
    print("\n" + "=" * 80)
    print("A/B TEST RESULTS")
    print("=" * 80)
    print()
    print(f"{'ID':<10} {'Baseline':>10} {'Muhasibi':>10} {'Diff':>10}")
    print("-" * 40)
    
    for q in results["questions"]:
        q_id = q["id"]
        b_score = q["baseline"]["score"]["average"]
        m_score = q["muhasibi"]["score"]["average"]
        diff = q["diff"]
        print(f"{q_id:<10} {b_score:>10.2f} {m_score:>10.2f} {diff:>+10.2f}")
    
    print("-" * 40)
    
    summary = results.get("summary", {})
    if summary:
        print(f"{'AVERAGE':<10} {summary['baseline_average']:>10.2f} {summary['muhasibi_average']:>10.2f} {summary['overall_difference']:>+10.2f}")
        print()
        print(f"Muhasibi adds value (diff >= +1.0): {summary['muhasibi_adds_value']}")


def main() -> None:
    """Main entry point."""
    load_dotenv()
    _require_env("DATABASE_URL")
    print("=" * 60)
    print("MUHASIBI A/B TEST HARNESS")
    print(f"Started at: {datetime.utcnow().isoformat()}")
    print("=" * 60)
    print()
    print("Using in-process FastAPI app (no external server).")
    print()
    
    results = asyncio.run(run_ab_test())
    
    # Write results
    output_path = Path("ab_test_results.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\nResults written to: {output_path}")
    
    # Print table
    print_results_table(results)
    
    # Exit code based on result
    summary = results.get("summary", {})
    if summary.get("muhasibi_adds_value"):
        print("\n✓ Muhasibi mode adds measurable value!")
        sys.exit(0)
    else:
        print("\n⚠ Muhasibi mode did not meet +1.0 improvement threshold")
        sys.exit(1)


if __name__ == "__main__":
    main()
