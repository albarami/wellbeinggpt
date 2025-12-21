"""
Run 160Q bakeoff with selective reranker mode.

This runs the full bakeoff dataset through the /ask/ui endpoint with:
- RERANKER_ENABLED=false
- RERANKER_SELECTIVE_MODE=true (per-intent policy gating)

Output:
- eval/reports/bakeoff_160q_selective.md
- eval/output/selective_bakeoff/bakeoff_selective_YYYYMMDD_HHMMSS.jsonl
"""

import json
import os
import sys
from datetime import datetime
from pathlib import Path
from collections import defaultdict

import requests

# Encoding fix for Windows
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

API_URL = "http://127.0.0.1:8000/ask/ui"
DATASET_PATH = Path("eval/datasets/bakeoff_depth_v1.jsonl")
OUTPUT_DIR = Path("eval/output/selective_bakeoff")
REPORT_PATH = Path("eval/reports/bakeoff_160q_selective.md")


def load_dataset():
    """Load the 160Q bakeoff dataset."""
    questions = []
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                questions.append(json.loads(line))
    return questions


def run_question(q: dict) -> dict:
    """Run a single question through the API."""
    qid = q.get("id") or q.get("qid", "?")
    question_text = q.get("question") or q.get("question_ar", "")
    qtype = q.get("type", "unknown")
    
    try:
        # Use the question type to determine mode
        # natural_chat questions should use mode=natural_chat to get proper routing
        request_mode = "natural_chat" if qtype == "natural_chat" else "answer"
        
        resp = requests.post(
            API_URL,
            json={"question": question_text, "debug": False, "mode": request_mode},
            timeout=120,
        )
        if resp.status_code != 200:
            return {
                "qid": qid,
                "type": qtype,
                "contract": "ERROR",
                "citations": 0,
                "used_edges": 0,
                "abstained": True,
                "error": f"HTTP {resp.status_code}",
            }
        
        data = resp.json()
        
        # /ask/ui uses citations_spans not citations
        citations = data.get("citations_spans", []) or data.get("citations", [])
        graph_trace = data.get("graph_trace", {}) or {}
        used_edges = graph_trace.get("used_edges", [])
        
        # Determine contract outcome
        contract = data.get("contract_outcome") or data.get("contract")
        if not contract:
            # Infer from structure
            abstain_reason = data.get("abstain_reason")
            if abstain_reason:
                contract = "FAIL"
            elif citations and len(citations) > 0:
                contract = "PASS_FULL"
            else:
                contract = "PASS_PARTIAL"
        
        # Retrieval debug info (now exposed in /ask/ui response)
        retrieval_debug = data.get("retrieval_debug", {}) or {}
        reranker_used = retrieval_debug.get("reranker_used", False)
        reranker_reason = retrieval_debug.get("reranker_reason", "unknown")
        seed_floor_applied = retrieval_debug.get("seed_floor_applied", False)
        bypass_relevance_gate = retrieval_debug.get("bypass_relevance_gate", False)
        not_found_reason = retrieval_debug.get("not_found_reason", "")
        
        return {
            "qid": qid,
            "type": qtype,
            "contract": contract,
            "citations": len(citations) if isinstance(citations, list) else 0,
            "used_edges": len(used_edges) if isinstance(used_edges, list) else 0,
            "abstained": bool(data.get("abstain_reason")),
            "reranker_used": reranker_used,
            "reranker_reason": reranker_reason,
            "seed_floor_applied": seed_floor_applied,
            "bypass_relevance_gate": bypass_relevance_gate,
            "not_found_reason": not_found_reason,
        }
    except Exception as e:
        import traceback
        print(f"  ERROR for {qid}: {e}")
        traceback.print_exc()
        return {
            "qid": qid,
            "type": qtype,
            "contract": "ERROR",
            "citations": 0,
            "used_edges": 0,
            "abstained": True,
            "error": str(e),
        }


def compute_metrics(results: list[dict]) -> dict:
    """Compute bakeoff metrics."""
    total = len(results)
    pass_full = sum(1 for r in results if r.get("contract") == "PASS_FULL")
    pass_partial = sum(1 for r in results if r.get("contract") == "PASS_PARTIAL")
    fail = sum(1 for r in results if r.get("contract") == "FAIL")
    errors = sum(1 for r in results if r.get("contract") == "ERROR")
    
    # Citation stats
    has_citations = sum(1 for r in results if r.get("citations", 0) > 0)
    
    # Reranker usage
    reranker_used = sum(1 for r in results if r.get("reranker_used"))
    
    # Seed floor stats
    seed_floor_applied = sum(1 for r in results if r.get("seed_floor_applied"))
    bypass_gate_applied = sum(1 for r in results if r.get("bypass_relevance_gate"))
    
    # By type
    by_type = defaultdict(lambda: {"total": 0, "pass_full": 0, "pass_partial": 0, "fail": 0, "reranker_used": 0})
    for r in results:
        qtype = r.get("type", "unknown")
        by_type[qtype]["total"] += 1
        if r.get("contract") == "PASS_FULL":
            by_type[qtype]["pass_full"] += 1
        elif r.get("contract") == "PASS_PARTIAL":
            by_type[qtype]["pass_partial"] += 1
        elif r.get("contract") == "FAIL":
            by_type[qtype]["fail"] += 1
        if r.get("reranker_used"):
            by_type[qtype]["reranker_used"] += 1
    
    # Expected failures (injection, out_of_scope)
    expected_fail_types = {"injection", "out_of_scope"}
    expected_fails = sum(1 for r in results if r.get("type") in expected_fail_types and r.get("contract") in ("FAIL", "PASS_PARTIAL"))
    unexpected_fails = fail - expected_fails
    
    # Cross-pillar edges
    cross_pillar_results = [r for r in results if r.get("type") == "cross_pillar"]
    mean_used_edges = (
        sum(r.get("used_edges", 0) for r in cross_pillar_results) / len(cross_pillar_results)
        if cross_pillar_results else 0.0
    )
    
    return {
        "total": total,
        "pass_full": pass_full,
        "pass_full_rate": pass_full / total if total > 0 else 0.0,
        "pass_partial": pass_partial,
        "seed_floor_applied_count": seed_floor_applied,
        "bypass_gate_applied_count": bypass_gate_applied,
        "fail": fail,
        "errors": errors,
        "unexpected_fail_rate": unexpected_fails / total if total > 0 else 0.0,
        "citation_present_rate": has_citations / total if total > 0 else 0.0,
        "mean_used_edges_cross_pillar": mean_used_edges,
        "reranker_used_count": reranker_used,
        "reranker_used_rate": reranker_used / total if total > 0 else 0.0,
        "by_type": dict(by_type),
    }


def write_report(metrics: dict, results: list[dict]):
    """Write markdown report."""
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("# 160Q Bakeoff - Selective Reranker Mode\n\n")
        f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n")
        f.write("**Configuration:**\n")
        f.write("- RERANKER_ENABLED=false\n")
        f.write("- RERANKER_SELECTIVE_MODE=true\n")
        f.write("- Model: checkpoints/reranker_phase2\n\n")
        
        f.write("## Overall Metrics\n\n")
        f.write(f"| Metric | Value |\n")
        f.write(f"|--------|-------|\n")
        f.write(f"| Total Questions | {metrics['total']} |\n")
        f.write(f"| **PASS_FULL Rate** | **{metrics['pass_full_rate']*100:.1f}%** |\n")
        f.write(f"| PASS_PARTIAL | {metrics['pass_partial']} |\n")
        f.write(f"| FAIL | {metrics['fail']} |\n")
        f.write(f"| **Unexpected Fail Rate** | **{metrics['unexpected_fail_rate']*100:.1f}%** |\n")
        f.write(f"| **Citation Present Rate** | **{metrics['citation_present_rate']*100:.1f}%** |\n")
        f.write(f"| **Mean Used Edges (Cross-Pillar)** | **{metrics['mean_used_edges_cross_pillar']:.1f}** |\n")
        f.write(f"| Reranker Used Count | {metrics['reranker_used_count']} |\n")
        f.write(f"| Reranker Used Rate | {metrics['reranker_used_rate']*100:.1f}% |\n\n")
        
        f.write("## Results by Question Type\n\n")
        f.write("| Type | Total | PASS_FULL | PASS_PARTIAL | FAIL | Reranker Used |\n")
        f.write("|------|-------|-----------|--------------|------|---------------|\n")
        
        for qtype, stats in sorted(metrics["by_type"].items()):
            pf_rate = stats["pass_full"] / stats["total"] * 100 if stats["total"] > 0 else 0
            rr_rate = stats["reranker_used"] / stats["total"] * 100 if stats["total"] > 0 else 0
            f.write(f"| {qtype} | {stats['total']} | {stats['pass_full']} ({pf_rate:.0f}%) | {stats['pass_partial']} | {stats['fail']} | {stats['reranker_used']} ({rr_rate:.0f}%) |\n")
        
        f.write("\n## Reranker Decision Distribution\n\n")
        decisions = defaultdict(int)
        for r in results:
            dec = r.get("reranker_decision", "unknown")
            decisions[dec] += 1
        
        f.write("| Decision | Count |\n")
        f.write("|----------|-------|\n")
        for dec, count in sorted(decisions.items(), key=lambda x: -x[1]):
            f.write(f"| {dec} | {count} |\n")
        
        # Failed questions
        failed = [r for r in results if r.get("contract") == "FAIL" and r.get("type") not in {"injection", "out_of_scope"}]
        if failed:
            f.write("\n## Unexpected Failures\n\n")
            for r in failed:
                f.write(f"- **{r['qid']}** ({r['type']}): cites={r['citations']}, edges={r['used_edges']}\n")
    
    print(f"\nReport written to: {REPORT_PATH}")


def main():
    print("=" * 60)
    print("160Q BAKEOFF - SELECTIVE RERANKER MODE")
    print("=" * 60)
    print(f"Dataset: {DATASET_PATH}")
    print()
    
    # Load dataset
    questions = load_dataset()
    print(f"Loaded {len(questions)} questions\n")
    
    # Run all questions
    results = []
    for i, q in enumerate(questions, 1):
        qid = q.get("id") or q.get("qid", "?")
        result = run_question(q)
        results.append(result)
        
        status = result.get("contract", "?")
        cites = result.get("citations", 0)
        edges = result.get("used_edges", 0)
        rr = "RR" if result.get("reranker_used") else "--"
        
        print(f"[{i}/{len(questions)}] {qid}... {status} (cites={cites}, edges={edges}) [{rr}]")
    
    # Save raw results
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = OUTPUT_DIR / f"bakeoff_selective_{timestamp}.jsonl"
    with open(output_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\nResults saved to: {output_path}")
    
    # Compute metrics
    metrics = compute_metrics(results)
    
    # Write report
    write_report(metrics, results)
    
    # Summary
    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"PASS_FULL Rate: {metrics['pass_full_rate']*100:.1f}%")
    print(f"Unexpected Fail Rate: {metrics['unexpected_fail_rate']*100:.1f}%")
    print(f"Citation Present Rate: {metrics['citation_present_rate']*100:.1f}%")
    print(f"Mean Used Edges (Cross-Pillar): {metrics['mean_used_edges_cross_pillar']:.1f}")
    print(f"Reranker Used: {metrics['reranker_used_count']}/{metrics['total']} ({metrics['reranker_used_rate']*100:.1f}%)")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
