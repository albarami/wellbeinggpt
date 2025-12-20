"""Reranker A/B Test: Compare system performance with reranker ON vs OFF.

This script runs the regression dataset through the API with:
- Reranker OFF (baseline)
- Reranker ON (treatment)

And measures:
- Contract PASS_FULL rate
- Citation count
- Used edges count
- Latency

Results are written to eval/reports/reranker_ab.md
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime
from pathlib import Path

import requests

REPO = Path(__file__).resolve().parents[1]
DATASET_PATH = REPO / "eval/datasets/regression_unexpected_fails.jsonl"
OUTPUT_DIR = REPO / "eval/output/reranker_ab"
REPORT_PATH = REPO / "eval/reports/reranker_ab.md"

API_URL = "http://127.0.0.1:8000/ask/ui"
DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-5.1")


def load_dataset() -> list[dict]:
    """Load regression dataset."""
    if not DATASET_PATH.exists():
        print(f"ERROR: {DATASET_PATH} not found")
        return []
    rows = []
    for line in DATASET_PATH.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def run_test(questions: list[dict], reranker_enabled: bool) -> dict:
    """Run test with reranker on or off.
    
    Note: The API needs to be restarted with RERANKER_ENABLED env var set appropriately.
    This script assumes the API is already running with the correct setting.
    """
    results = []
    total_latency = 0.0
    
    label = "ON" if reranker_enabled else "OFF"
    print(f"\n{'='*60}")
    print(f"Running with Reranker {label}")
    print(f"{'='*60}")
    
    for i, q in enumerate(questions):
        qid = q.get("id", f"q{i}")
        question_ar = q.get("question_ar", "")
        qtype = q.get("type", "generic")
        mode = "natural_chat" if qtype == "natural_chat" else "answer"
        
        try:
            start = time.time()
            resp = requests.post(
                API_URL,
                json={
                    "question": question_ar,
                    "mode": mode,
                    "model_deployment": DEPLOYMENT,
                },
                timeout=120,
            )
            elapsed = time.time() - start
            total_latency += elapsed
            
            if resp.status_code != 200:
                results.append({
                    "id": qid,
                    "success": False,
                    "contract_outcome": "ERROR",
                    "citations_count": 0,
                    "edges_count": 0,
                    "latency": elapsed,
                    "error": f"HTTP {resp.status_code}",
                })
                print(f"[{i+1}/{len(questions)}] {qid}: ERROR (HTTP {resp.status_code})")
                continue
            
            data = resp.json()
            outcome = data.get("contract_outcome", "UNKNOWN")
            cites = len(data.get("citations_spans") or data.get("citations") or [])
            edges = len(data.get("used_edges") or [])
            abstained = data.get("abstained", False)
            
            success = outcome in ("PASS_FULL", "PASS_PARTIAL") and not abstained
            
            results.append({
                "id": qid,
                "success": success,
                "contract_outcome": outcome,
                "citations_count": cites,
                "edges_count": edges,
                "latency": elapsed,
            })
            
            status = "PASS" if success else "FAIL"
            print(f"[{i+1}/{len(questions)}] {qid}: {status} ({outcome}, cites={cites}, edges={edges}, {elapsed:.1f}s)")
            
        except Exception as e:
            results.append({
                "id": qid,
                "success": False,
                "contract_outcome": "ERROR",
                "citations_count": 0,
                "edges_count": 0,
                "latency": 0,
                "error": str(e),
            })
            print(f"[{i+1}/{len(questions)}] {qid}: ERROR ({e})")
    
    # Compute summary
    total = len(results)
    passed = sum(1 for r in results if r["success"])
    pass_full = sum(1 for r in results if r["contract_outcome"] == "PASS_FULL")
    pass_partial = sum(1 for r in results if r["contract_outcome"] == "PASS_PARTIAL")
    avg_cites = sum(r["citations_count"] for r in results) / total if total else 0
    avg_edges = sum(r["edges_count"] for r in results) / total if total else 0
    avg_latency = total_latency / total if total else 0
    
    return {
        "reranker_enabled": reranker_enabled,
        "total": total,
        "passed": passed,
        "pass_rate": passed / total if total else 0,
        "pass_full": pass_full,
        "pass_partial": pass_partial,
        "avg_citations": avg_cites,
        "avg_edges": avg_edges,
        "avg_latency": avg_latency,
        "results": results,
    }


def generate_report(off_results: dict, on_results: dict) -> str:
    """Generate markdown report."""
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    
    report = f"""# Reranker A/B Test Report

Generated: {ts}
Deployment: {DEPLOYMENT}
Dataset: {DATASET_PATH.name}

## Summary

| Metric | Reranker OFF | Reranker ON | Delta |
|--------|--------------|-------------|-------|
| Pass Rate | {off_results['pass_rate']:.1%} | {on_results['pass_rate']:.1%} | {(on_results['pass_rate'] - off_results['pass_rate']) * 100:+.1f}pp |
| PASS_FULL | {off_results['pass_full']}/{off_results['total']} | {on_results['pass_full']}/{on_results['total']} | {on_results['pass_full'] - off_results['pass_full']:+d} |
| Avg Citations | {off_results['avg_citations']:.1f} | {on_results['avg_citations']:.1f} | {on_results['avg_citations'] - off_results['avg_citations']:+.1f} |
| Avg Edges | {off_results['avg_edges']:.1f} | {on_results['avg_edges']:.1f} | {on_results['avg_edges'] - off_results['avg_edges']:+.1f} |
| Avg Latency | {off_results['avg_latency']:.1f}s | {on_results['avg_latency']:.1f}s | {on_results['avg_latency'] - off_results['avg_latency']:+.1f}s |

## Analysis

"""
    
    # Determine winner
    if on_results['pass_rate'] > off_results['pass_rate']:
        report += "**Reranker improves pass rate.** Recommend keeping reranker ON.\n\n"
    elif on_results['pass_rate'] < off_results['pass_rate']:
        report += "**Reranker reduces pass rate.** Investigate before enabling.\n\n"
    else:
        if on_results['avg_citations'] > off_results['avg_citations']:
            report += "**Pass rate equal, but reranker increases citations.** Recommend keeping reranker ON.\n\n"
        else:
            report += "**No significant difference.** Reranker is neutral.\n\n"
    
    # Latency impact
    if on_results['avg_latency'] > off_results['avg_latency'] * 1.2:
        report += f"Note: Reranker adds {on_results['avg_latency'] - off_results['avg_latency']:.1f}s latency.\n\n"
    
    report += "## Per-Question Comparison\n\n"
    report += "| Question | OFF | ON | Delta |\n"
    report += "|----------|-----|-----|-------|\n"
    
    for i in range(len(off_results['results'])):
        off_r = off_results['results'][i]
        on_r = on_results['results'][i]
        qid = off_r['id']
        
        off_status = f"{off_r['contract_outcome']} ({off_r['citations_count']}c)"
        on_status = f"{on_r['contract_outcome']} ({on_r['citations_count']}c)"
        
        delta_cites = on_r['citations_count'] - off_r['citations_count']
        delta = f"{delta_cites:+d}c" if delta_cites != 0 else "="
        
        report += f"| {qid} | {off_status} | {on_status} | {delta} |\n"
    
    report += "\n## Conclusion\n\n"
    if on_results['pass_rate'] >= off_results['pass_rate'] and on_results['avg_citations'] >= off_results['avg_citations']:
        report += "Reranker provides value. Keep it enabled.\n"
    else:
        report += "Reranker value unclear. Run on larger dataset to confirm.\n"
    
    return report


def restart_api_with_reranker(enabled: bool) -> bool:
    """Restart API with reranker enabled or disabled."""
    import subprocess
    import time as time_module
    
    # Kill existing API
    subprocess.run(["taskkill", "/F", "/IM", "python.exe"], capture_output=True)
    time_module.sleep(2)
    
    # Set environment and start API
    env = os.environ.copy()
    env["RERANKER_ENABLED"] = "true" if enabled else "false"
    
    proc = subprocess.Popen(
        ["python", "-m", "uvicorn", "apps.api.main:app", "--host", "127.0.0.1", "--port", "8000"],
        cwd=REPO,
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    
    # Wait for API to be ready
    for _ in range(30):
        time_module.sleep(1)
        try:
            resp = requests.get("http://127.0.0.1:8000/health", timeout=2)
            if resp.status_code == 200:
                return True
        except Exception:
            pass
    
    # Try anyway
    return True


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    questions = load_dataset()
    if not questions:
        return
    
    print(f"Loaded {len(questions)} questions")
    print(f"Deployment: {DEPLOYMENT}")
    
    # Run with reranker OFF
    print("\nRestarting API with reranker OFF...")
    restart_api_with_reranker(enabled=False)
    off_results = run_test(questions, reranker_enabled=False)
    
    # Run with reranker ON
    print("\nRestarting API with reranker ON...")
    restart_api_with_reranker(enabled=True)
    on_results = run_test(questions, reranker_enabled=True)
    
    # Save raw results
    (OUTPUT_DIR / "reranker_off.json").write_text(
        json.dumps(off_results, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )
    (OUTPUT_DIR / "reranker_on.json").write_text(
        json.dumps(on_results, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )
    
    # Generate report
    report = generate_report(off_results, on_results)
    REPORT_PATH.write_text(report, encoding="utf-8")
    
    print(f"\n{'='*60}")
    print("RERANKER A/B TEST COMPLETE")
    print(f"{'='*60}")
    print(f"OFF: {off_results['pass_rate']:.1%} pass, {off_results['avg_citations']:.1f} avg cites")
    print(f"ON:  {on_results['pass_rate']:.1%} pass, {on_results['avg_citations']:.1f} avg cites")
    print(f"\nReport: {REPORT_PATH}")


if __name__ == "__main__":
    main()
