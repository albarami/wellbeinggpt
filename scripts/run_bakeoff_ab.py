"""
Run 160-question bakeoff with reranker ON vs OFF.
Computes deltas on depth/quality metrics.
"""
import sys
import json
import time
import subprocess
import os
import signal
from pathlib import Path
from datetime import datetime
from typing import Optional

REPO = Path(__file__).resolve().parents[1]
DATASET = REPO / "eval/datasets/bakeoff_depth_v1.jsonl"
OUT_DIR = REPO / "eval/output/bakeoff_ab"
REPORT_PATH = REPO / "eval/reports/reranker_ab_bakeoff_depth.md"

API_URL = "http://127.0.0.1:8000"
DEPLOYMENT = "gpt-5.1"

OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_dataset():
    rows = []
    with DATASET.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                rows.append(json.loads(s))
    return rows


def run_single_question(question: str, mode: str, timeout: int = 180):
    """Run a single question and return detailed result."""
    import requests
    try:
        response = requests.post(
            f"{API_URL}/ask/ui",
            json={
                "question": question,
                "model_deployment": DEPLOYMENT,
                "mode": mode,
            },
            timeout=timeout,
        )
        if response.status_code == 200:
            data = response.json()
            
            # Extract metrics
            citations = data.get("citations_spans") or []
            graph_trace = data.get("graph_trace") or {}
            used_edges = graph_trace.get("used_edges") or []
            argument_chains = graph_trace.get("argument_chains") or []
            final = data.get("final") or {}
            answer = final.get("answer_ar") or ""
            
            # Check for boundary section
            has_boundary = "حدود" in answer or "غير منصوص" in answer
            
            return {
                "contract_outcome": data.get("contract_outcome"),
                "citations_count": len(citations),
                "used_edges_count": len(used_edges),
                "argument_chains_count": len(argument_chains),
                "answer_length": len(answer),
                "has_boundary": has_boundary,
                "abstained": data.get("abstain_reason") is not None,
                "success": data.get("contract_outcome") in ("PASS_FULL", "PASS_PARTIAL"),
                "pass_full": data.get("contract_outcome") == "PASS_FULL",
            }
        return {"error": f"HTTP {response.status_code}", "success": False, "pass_full": False}
    except Exception as e:
        return {"error": str(e), "success": False, "pass_full": False}


def run_bakeoff(rows, condition: str):
    """Run full bakeoff and return results."""
    print(f"\n{'='*60}")
    print(f"Running bakeoff: {condition}")
    print(f"Questions: {len(rows)}")
    print(f"{'='*60}\n")
    
    results = []
    
    for i, row in enumerate(rows):
        qid = row.get("id", f"q{i}")
        question = row.get("question_ar") or row.get("question", "")
        qtype = row.get("type", "")
        mode = "natural_chat" if qtype == "natural_chat" else "answer"
        
        result = run_single_question(question, mode)
        result["id"] = qid
        result["type"] = qtype
        results.append(result)
        
        status = "PASS" if result.get("success") else "FAIL"
        cites = result.get("citations_count", 0)
        edges = result.get("used_edges_count", 0)
        
        # Print progress every 10 questions
        if (i + 1) % 10 == 0 or i == len(rows) - 1:
            passed = sum(1 for r in results if r.get("success"))
            print(f"  [{i+1}/{len(rows)}] {passed} passed so far...")
    
    return results


def compute_metrics(results):
    """Compute aggregate metrics from results."""
    total = len(results)
    if total == 0:
        return {}
    
    passed = sum(1 for r in results if r.get("success"))
    pass_full = sum(1 for r in results if r.get("pass_full"))
    with_citations = sum(1 for r in results if r.get("citations_count", 0) > 0)
    with_edges = sum(1 for r in results if r.get("used_edges_count", 0) > 0)
    with_chains = sum(1 for r in results if r.get("argument_chains_count", 0) > 0)
    with_boundary = sum(1 for r in results if r.get("has_boundary"))
    
    citations = [r.get("citations_count", 0) for r in results]
    edges = [r.get("used_edges_count", 0) for r in results]
    chains = [r.get("argument_chains_count", 0) for r in results]
    
    return {
        "total": total,
        "passed": passed,
        "pass_rate": passed / total,
        "pass_full": pass_full,
        "pass_full_rate": pass_full / total,
        "citation_present_rate": with_citations / total,
        "mean_citations": sum(citations) / total,
        "mean_edges": sum(edges) / total,
        "mean_chains": sum(chains) / total,
        "boundary_rate": with_boundary / total,
        "unexpected_fail_rate": (total - passed) / total,
    }


def wait_for_api() -> bool:
    """Wait for API to be ready."""
    import requests
    print("  Waiting for API...")
    sys.stdout.flush()
    for attempt in range(30):
        try:
            r = requests.post(
                f"{API_URL}/ask/ui",
                json={"question": "test", "mode": "answer", "model_deployment": DEPLOYMENT},
                timeout=60
            )
            if r.status_code == 200:
                print("  API ready!")
                sys.stdout.flush()
                return True
        except Exception as e:
            if attempt == 0:
                print(f"  Connecting... ({e})")
                sys.stdout.flush()
        time.sleep(2)
    
    print("  API not responding!")
    return False


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", choices=["off", "on", "both"], default="both",
                       help="Which condition to run")
    args = parser.parse_args()
    
    print("=" * 60)
    print("RERANKER A/B BAKEOFF TEST")
    print(f"Dataset: {DATASET}")
    print(f"Deployment: {DEPLOYMENT}")
    print(f"Condition: {args.condition}")
    print("=" * 60)
    sys.stdout.flush()
    
    rows = load_dataset()
    print(f"Loaded {len(rows)} questions")
    sys.stdout.flush()
    
    metrics_off = None
    metrics_on = None
    results_off = []
    results_on = []
    
    # Load existing results if running single condition
    if args.condition == "on" and (OUT_DIR / "reranker_off_metrics.json").exists():
        metrics_off = json.loads((OUT_DIR / "reranker_off_metrics.json").read_text())
        print(f"Loaded existing OFF metrics: {metrics_off['pass_full_rate']*100:.1f}% PASS_FULL")
    
    if args.condition == "off" and (OUT_DIR / "reranker_on_metrics.json").exists():
        metrics_on = json.loads((OUT_DIR / "reranker_on_metrics.json").read_text())
        print(f"Loaded existing ON metrics: {metrics_on['pass_full_rate']*100:.1f}% PASS_FULL")
    
    if args.condition in ("off", "both"):
        # Expect API to be running with reranker OFF
        print("\nEnsure API is running with RERANKER_ENABLED=false")
        sys.stdout.flush()
        if not wait_for_api():
            print("API not responding. Start it with: $env:RERANKER_ENABLED='false'; python -m uvicorn apps.api.main:app")
            return 1
        
        results_off = run_bakeoff(rows, "RERANKER OFF")
        metrics_off = compute_metrics(results_off)
        
        # Save OFF results
        (OUT_DIR / "reranker_off.jsonl").write_text(
            "\n".join(json.dumps(r, ensure_ascii=False) for r in results_off),
            encoding="utf-8"
        )
        (OUT_DIR / "reranker_off_metrics.json").write_text(
            json.dumps(metrics_off, ensure_ascii=False, indent=2),
            encoding="utf-8"
        )
        print(f"\nOFF complete: {metrics_off['pass_full_rate']*100:.1f}% PASS_FULL")
        sys.stdout.flush()
        
        if args.condition == "off":
            print("\nRun with --condition=on after restarting API with RERANKER_ENABLED=true")
            return 0
    
    if args.condition in ("on", "both"):
        if args.condition == "both":
            print("\n*** NOW RESTART API WITH RERANKER_ENABLED=true ***")
            print("Press Enter when ready...")
            sys.stdout.flush()
            input()
        
        print("\nEnsure API is running with RERANKER_ENABLED=true")
        sys.stdout.flush()
        if not wait_for_api():
            print("API not responding. Start it with: $env:RERANKER_ENABLED='true'; python -m uvicorn apps.api.main:app")
            return 1
        
        results_on = run_bakeoff(rows, "RERANKER ON")
        metrics_on = compute_metrics(results_on)
    
        # Save ON results
        (OUT_DIR / "reranker_on.jsonl").write_text(
            "\n".join(json.dumps(r, ensure_ascii=False) for r in results_on),
            encoding="utf-8"
        )
        (OUT_DIR / "reranker_on_metrics.json").write_text(
            json.dumps(metrics_on, ensure_ascii=False, indent=2),
            encoding="utf-8"
        )
        print(f"\nON complete: {metrics_on['pass_full_rate']*100:.1f}% PASS_FULL")
        sys.stdout.flush()
    
    # If we don't have both metrics, can't generate comparison report
    if metrics_off is None or metrics_on is None:
        print("\nNeed both conditions to generate comparison report.")
        return 0
    
    # Compute deltas
    deltas = {}
    for key in metrics_on:
        if isinstance(metrics_on[key], (int, float)) and isinstance(metrics_off.get(key), (int, float)):
            deltas[key] = metrics_on[key] - metrics_off[key]
    
    # Generate report
    report = f"""# Reranker A/B Bakeoff Test

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}
Dataset: bakeoff_depth_v1.jsonl ({len(rows)} questions)
Deployment: {DEPLOYMENT}

## Summary

| Metric | Reranker OFF | Reranker ON | Delta |
|--------|--------------|-------------|-------|
| PASS_FULL Rate | {metrics_off['pass_full_rate']*100:.1f}% | {metrics_on['pass_full_rate']*100:.1f}% | {deltas['pass_full_rate']*100:+.1f}% |
| Pass Rate (any) | {metrics_off['pass_rate']*100:.1f}% | {metrics_on['pass_rate']*100:.1f}% | {deltas['pass_rate']*100:+.1f}% |
| Citation Present Rate | {metrics_off['citation_present_rate']*100:.1f}% | {metrics_on['citation_present_rate']*100:.1f}% | {deltas['citation_present_rate']*100:+.1f}% |
| Mean Citations | {metrics_off['mean_citations']:.1f} | {metrics_on['mean_citations']:.1f} | {deltas['mean_citations']:+.1f} |
| Mean Used Edges | {metrics_off['mean_edges']:.1f} | {metrics_on['mean_edges']:.1f} | {deltas['mean_edges']:+.1f} |
| Mean Argument Chains | {metrics_off['mean_chains']:.1f} | {metrics_on['mean_chains']:.1f} | {deltas['mean_chains']:+.1f} |
| Boundary Rate | {metrics_off['boundary_rate']*100:.1f}% | {metrics_on['boundary_rate']*100:.1f}% | {deltas['boundary_rate']*100:+.1f}% |
| Unexpected Fail Rate | {metrics_off['unexpected_fail_rate']*100:.1f}% | {metrics_on['unexpected_fail_rate']*100:.1f}% | {deltas['unexpected_fail_rate']*100:+.1f}% |

## Detailed Counts

| Condition | Total | Passed | PASS_FULL | Failed |
|-----------|-------|--------|-----------|--------|
| Reranker OFF | {metrics_off['total']} | {metrics_off['passed']} | {metrics_off['pass_full']} | {metrics_off['total'] - metrics_off['passed']} |
| Reranker ON | {metrics_on['total']} | {metrics_on['passed']} | {metrics_on['pass_full']} | {metrics_on['total'] - metrics_on['passed']} |

## Interpretation

"""
    
    # Interpret results
    if deltas['pass_full_rate'] > 0.02:
        report += f"**Reranker improves PASS_FULL rate by {deltas['pass_full_rate']*100:.1f}%** - significant uplift.\n\n"
    elif deltas['pass_full_rate'] < -0.02:
        report += f"**Reranker DECREASES PASS_FULL rate by {abs(deltas['pass_full_rate'])*100:.1f}%** - unexpected regression.\n\n"
    else:
        report += "**PASS_FULL rate is similar** with and without reranker.\n\n"
    
    if deltas['mean_citations'] > 0.5:
        report += f"**Reranker increases mean citations by {deltas['mean_citations']:.1f}** - better evidence retrieval.\n\n"
    
    if deltas['mean_edges'] > 0.2:
        report += f"**Reranker increases mean used edges by {deltas['mean_edges']:.1f}** - improved graph utilization.\n\n"
    
    # Recommendation
    report += """## Recommendation

"""
    
    if deltas['pass_full_rate'] > 0 and deltas['unexpected_fail_rate'] <= 0:
        report += """**Enable reranker in production.**

The reranker provides:
- Higher PASS_FULL rate
- No increase in unexpected failures
- Better evidence ranking for synthesis questions
"""
    elif deltas['pass_full_rate'] > 0 and deltas['unexpected_fail_rate'] > 0:
        report += """**Enable reranker selectively.**

The reranker improves PASS_FULL but also increases some failures.
Recommend: Enable only for global_synthesis and cross_pillar intents.
"""
    else:
        report += """**Keep reranker in staging only.**

No clear improvement on the full bakeoff.
Consider: Training on more data or adjusting reranker threshold.
"""
    
    # Write report
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(report, encoding="utf-8")
    print(f"\nReport written to: {REPORT_PATH}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
