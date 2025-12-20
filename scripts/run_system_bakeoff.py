"""
System Bakeoff - Test the FULL Muḥāsibī system with different models
=====================================================================
This tests the actual system (retrieval, graph, integrity, citations)
with gpt-5-chat, gpt-5.1, and gpt-5.2 to see which model works best.

Usage:
    1. Start the API server: python -m apps.api.main
    2. Run this script: python scripts/run_system_bakeoff.py
"""

import os
import sys
import time
import json
import requests
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

# Config
API_BASE = os.getenv("API_BASE_URL", "http://localhost:8000")
DEPLOYMENTS = ["gpt-5-chat", "gpt-5.1", "gpt-5.2"]

# 30 depth-focused questions for quick comparison
QUESTIONS = [
    # 10 synthesis (global/world-model)
    {"id": "s01", "q": "كيف يؤدي الإطار إلى ازدهار الإنسان؟", "cat": "synthesis"},
    {"id": "s02", "q": "ما العلاقة بين جميع الركائز الخمس؟", "cat": "synthesis"},
    {"id": "s03", "q": "كيف تعمل الركائز معًا لتحقيق التوازن؟", "cat": "synthesis"},
    {"id": "s04", "q": "ما خطوات تحقيق الحياة الطيبة وفق الإطار؟", "cat": "synthesis"},
    {"id": "s05", "q": "ما المخاطر من إهمال إحدى الركائز؟", "cat": "synthesis"},
    {"id": "s06", "q": "ما الآليات الداخلية التي تربط الركائز ببعضها؟", "cat": "synthesis"},
    {"id": "s07", "q": "كيف يبني الإطار نموذجاً متكاملاً للإنسان؟", "cat": "synthesis"},
    {"id": "s08", "q": "ما الحلقات السببية بين الركائز الخمس؟", "cat": "synthesis"},
    {"id": "s09", "q": "كيف تتفاعل الحلقات السببية في نظام الركائز؟", "cat": "synthesis"},
    {"id": "s10", "q": "ما دور كل ركيزة في تحقيق الازدهار الشامل؟", "cat": "synthesis"},
    # 10 cross-pillar
    {"id": "c01", "q": "كيف ترتبط الحياة الروحية بالحياة الاجتماعية؟", "cat": "cross"},
    {"id": "c02", "q": "ما علاقة الإحسان بالتوكل؟", "cat": "cross"},
    {"id": "c03", "q": "كيف يؤثر التوازن الروحي على الصحة البدنية؟", "cat": "cross"},
    {"id": "c04", "q": "ما تأثير العلاقات الاجتماعية على الصحة النفسية؟", "cat": "cross"},
    {"id": "c05", "q": "كيف يعزز الإيمان الصحة العاطفية؟", "cat": "cross"},
    {"id": "c06", "q": "اربط بين الاستقامة والصحة الفكرية", "cat": "cross"},
    {"id": "c07", "q": "ما العلاقة بين الحكمة والعلاقات الاجتماعية؟", "cat": "cross"},
    {"id": "c08", "q": "كيف يرتبط الصبر بالصحة الجسدية؟", "cat": "cross"},
    {"id": "c09", "q": "ما العلاقة بين الشكر والسعادة النفسية؟", "cat": "cross"},
    {"id": "c10", "q": "كيف تؤثر الطمأنينة النفسية على الصحة الجسدية؟", "cat": "cross"},
    # 5 boundaries
    {"id": "b01", "q": "ما الفرق بين الصبر والرضا؟", "cat": "boundaries"},
    {"id": "b02", "q": "ما الفرق بين الإيمان والتوكل من حيث الأثر العملي؟", "cat": "boundaries"},
    {"id": "b03", "q": "ما حدود التوكل مع الأخذ بالأسباب؟", "cat": "boundaries"},
    {"id": "b04", "q": "فرّق بين العبادة والإيمان داخل الإطار", "cat": "boundaries"},
    {"id": "b05", "q": "ما التناقضات المحتملة بين الركائز وكيف تُحل؟", "cat": "boundaries"},
    # 5 natural chat
    {"id": "n01", "q": "أشعر بالضيق هذه الأيام، ماذا يقول الإطار عن ذلك؟", "cat": "chat"},
    {"id": "n02", "q": "كيف أتعامل مع القلق من وجهة نظر الإطار؟", "cat": "chat"},
    {"id": "n03", "q": "كيف أطبق الصبر عند الغضب؟", "cat": "chat"},
    {"id": "n04", "q": "كيف أقوي إيماني؟", "cat": "chat"},
    {"id": "n05", "q": "كيف أوازن بين عملي وحياتي الروحية؟", "cat": "chat"},
]


@dataclass
class QuestionResult:
    qid: str
    category: str
    success: bool
    words: int = 0
    citations: int = 0
    pillars: int = 0
    edges: int = 0
    chains: int = 0
    contract_outcome: str = ""
    latency_ms: int = 0
    error: str = ""


@dataclass
class ModelStats:
    deployment: str
    results: list = field(default_factory=list)
    
    @property
    def success_rate(self) -> float:
        if not self.results:
            return 0
        return sum(1 for r in self.results if r.success) / len(self.results) * 100
    
    @property
    def avg_words(self) -> float:
        ok = [r for r in self.results if r.success]
        return sum(r.words for r in ok) / max(1, len(ok))
    
    @property
    def avg_citations(self) -> float:
        ok = [r for r in self.results if r.success]
        return sum(r.citations for r in ok) / max(1, len(ok))
    
    @property
    def avg_pillars(self) -> float:
        ok = [r for r in self.results if r.success]
        return sum(r.pillars for r in ok) / max(1, len(ok))
    
    @property
    def avg_edges(self) -> float:
        ok = [r for r in self.results if r.success]
        return sum(r.edges for r in ok) / max(1, len(ok))
    
    @property
    def avg_chains(self) -> float:
        ok = [r for r in self.results if r.success]
        return sum(r.chains for r in ok) / max(1, len(ok))
    
    @property
    def pass_full_rate(self) -> float:
        ok = [r for r in self.results if r.success]
        return sum(1 for r in ok if r.contract_outcome == "PASS_FULL") / max(1, len(ok)) * 100
    
    def depth_score(self) -> float:
        """Score based on words, citations, structure."""
        return (
            min(self.avg_words / 400, 1) * 30 +
            min(self.avg_citations / 5, 1) * 40 +
            (self.pass_full_rate / 100) * 30
        )
    
    def cross_pillar_score(self) -> float:
        """Score based on pillars, edges, chains."""
        return (
            min(self.avg_pillars / 4, 1) * 40 +
            min(self.avg_edges / 5, 1) * 35 +
            min(self.avg_chains / 3, 1) * 25
        )
    
    def composite_score(self) -> float:
        """Weighted composite: 45% depth + 35% cross-pillar + 20% reliability."""
        return (
            self.depth_score() * 0.45 +
            self.cross_pillar_score() * 0.35 +
            self.success_rate * 0.20
        )


def call_ask_endpoint(question: str, deployment: str) -> dict:
    """Call the /ask endpoint with specific model deployment."""
    try:
        # The system should use the deployment via environment or config
        # We'll pass it as a header or query param if supported
        resp = requests.post(
            f"{API_BASE}/ask",
            json={
                "question": question,
                "mode": "answer",
                "model_deployment": deployment,  # If API supports this
            },
            headers={
                "X-Model-Deployment": deployment,  # Alternative header
            },
            timeout=600
        )
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        return {"error": str(e)}


def extract_metrics(response: dict) -> dict:
    """Extract quality metrics from API response."""
    if "error" in response:
        return {"success": False, "error": response["error"]}
    
    try:
        # Extract from response structure
        answer = response.get("answer", "")
        citations = response.get("citations", [])
        used_edges = response.get("used_edges", [])
        argument_chains = response.get("argument_chains", [])
        contract_outcome = response.get("contract_outcome", "")
        pillars_covered = response.get("pillars_covered", [])
        
        # Count distinct pillars from citations/edges
        pillar_set = set(pillars_covered) if pillars_covered else set()
        if not pillar_set and used_edges:
            for edge in used_edges:
                if isinstance(edge, dict):
                    pillar_set.add(edge.get("from_pillar", ""))
                    pillar_set.add(edge.get("to_pillar", ""))
            pillar_set.discard("")
        
        return {
            "success": True,
            "words": len(answer.split()) if answer else 0,
            "citations": len(citations) if citations else 0,
            "pillars": len(pillar_set),
            "edges": len(used_edges) if used_edges else 0,
            "chains": len(argument_chains) if argument_chains else 0,
            "contract_outcome": contract_outcome or "UNKNOWN",
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


def run_bakeoff():
    print("=" * 70)
    print("SYSTEM BAKEOFF - Testing Full Muhasibi System")
    print("=" * 70)
    print(f"API: {API_BASE}")
    print(f"Models: {DEPLOYMENTS}")
    print(f"Questions: {len(QUESTIONS)}")
    print()
    
    # Check API is running
    try:
        health = requests.get(f"{API_BASE}/health", timeout=5)
        print(f"API Health: {health.status_code}")
    except Exception as e:
        print(f"ERROR: API not running at {API_BASE}")
        print(f"Start it with: python -m apps.api.main")
        return
    
    all_stats = []
    
    for dep in DEPLOYMENTS:
        print(f"\n{'-'*70}")
        print(f"TESTING: {dep}")
        print(f"{'-'*70}")
        
        stats = ModelStats(deployment=dep)
        
        for i, q in enumerate(QUESTIONS):
            print(f"  [{i+1}/{len(QUESTIONS)}] {q['id']}: {q['q'][:35]}... ", end="", flush=True)
            
            t0 = time.perf_counter()
            response = call_ask_endpoint(q["q"], dep)
            latency_ms = int((time.perf_counter() - t0) * 1000)
            
            metrics = extract_metrics(response)
            
            result = QuestionResult(
                qid=q["id"],
                category=q["cat"],
                success=metrics.get("success", False),
                words=metrics.get("words", 0),
                citations=metrics.get("citations", 0),
                pillars=metrics.get("pillars", 0),
                edges=metrics.get("edges", 0),
                chains=metrics.get("chains", 0),
                contract_outcome=metrics.get("contract_outcome", ""),
                latency_ms=latency_ms,
                error=metrics.get("error", ""),
            )
            stats.results.append(result)
            
            if result.success:
                print(f"OK ({latency_ms}ms) w={result.words} c={result.citations} p={result.pillars} e={result.edges}")
            else:
                print(f"ERROR: {result.error[:50]}")
        
        all_stats.append(stats)
    
    # Results
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    
    ranked = sorted(all_stats, key=lambda s: -s.composite_score())
    
    print(f"\n{'Model':<15} {'Composite':>10} {'Depth':>8} {'Cross':>8} {'Success':>8} {'Words':>8} {'Cites':>6} {'Pillars':>8} {'Edges':>6}")
    print("-" * 95)
    for s in ranked:
        print(f"{s.deployment:<15} {s.composite_score():>10.1f} {s.depth_score():>8.1f} {s.cross_pillar_score():>8.1f} {s.success_rate:>7.0f}% {s.avg_words:>8.0f} {s.avg_citations:>6.1f} {s.avg_pillars:>8.2f} {s.avg_edges:>6.1f}")
    
    winner = ranked[0]
    print(f"\n*** WINNER: {winner.deployment} (composite={winner.composite_score():.1f}) ***")
    
    # Write report
    Path("eval/reports").mkdir(parents=True, exist_ok=True)
    with open("eval/reports/model_bakeoff_depth.md", "w", encoding="utf-8") as f:
        f.write("# System Bakeoff Report\n\n")
        f.write(f"Generated: {datetime.now().isoformat()}\n\n")
        f.write("## Configuration\n\n")
        f.write("- **Test**: Full Muḥāsibī system (retrieval + graph + integrity + citations)\n")
        f.write(f"- **Questions**: {len(QUESTIONS)} depth-focused questions\n")
        f.write("- **Scoring**: 45% Depth + 35% Cross-pillar + 20% Reliability\n\n")
        f.write("## Results\n\n")
        f.write("| Model | Composite | Depth | Cross-Pillar | Success% | Avg Words | Avg Cites | Avg Pillars | Avg Edges |\n")
        f.write("|-------|-----------|-------|--------------|----------|-----------|-----------|-------------|----------|\n")
        for s in ranked:
            f.write(f"| {s.deployment} | {s.composite_score():.1f} | {s.depth_score():.1f} | {s.cross_pillar_score():.1f} | {s.success_rate:.0f}% | {s.avg_words:.0f} | {s.avg_citations:.1f} | {s.avg_pillars:.2f} | {s.avg_edges:.1f} |\n")
        f.write(f"\n## Winner\n\n**{winner.deployment}** with composite score {winner.composite_score():.1f}\n")
        
        f.write("\n## Metrics Explained\n\n")
        f.write("- **Depth Score**: Words (30%) + Citations (40%) + PASS_FULL rate (30%)\n")
        f.write("- **Cross-Pillar Score**: Pillars covered (40%) + Edges used (35%) + Argument chains (25%)\n")
        f.write("- **Composite**: 45% Depth + 35% Cross-pillar + 20% Success rate\n")
    
    print(f"\nReport: eval/reports/model_bakeoff_depth.md")


if __name__ == "__main__":
    run_bakeoff()


