"""
Run regression suite N times with reranker OFF to measure stability.
Proves whether synth-006 failures are transient or consistent.
"""
import sys
import json
import time
import subprocess
import os
from pathlib import Path
from datetime import datetime

REPO = Path(__file__).resolve().parents[1]
DATASET = REPO / "eval/datasets/regression_unexpected_fails.jsonl"
OUT_DIR = REPO / "eval/output/stability"
REPORT_PATH = REPO / "eval/reports/reranker_regression_stability.md"

N_RUNS = 10
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


def run_single_question(question: str, mode: str):
    """Run a single question and return result."""
    import requests
    try:
        response = requests.post(
            f"{API_URL}/ask/ui",
            json={
                "question": question,
                "model_deployment": DEPLOYMENT,
                "mode": mode,
            },
            timeout=180,
        )
        if response.status_code == 200:
            data = response.json()
            return {
                "contract_outcome": data.get("contract_outcome"),
                "citations_count": len(data.get("citations_spans") or []),
                "success": data.get("contract_outcome") in ("PASS_FULL", "PASS_PARTIAL"),
            }
        return {"error": f"HTTP {response.status_code}", "success": False}
    except Exception as e:
        return {"error": str(e), "success": False}


def run_regression_once(rows, run_id: int):
    """Run full regression suite once."""
    results = {}
    for row in rows:
        qid = row.get("id", "unknown")
        question = row.get("question_ar") or row.get("question", "")
        qtype = row.get("type", "")
        mode = "natural_chat" if qtype == "natural_chat" else "answer"
        
        result = run_single_question(question, mode)
        results[qid] = result
        
        status = "PASS" if result.get("success") else "FAIL"
        cites = result.get("citations_count", 0)
        print(f"  [{qid}] {status} (cites={cites})")
    
    return results


def main():
    print("=" * 60)
    print("RERANKER STABILITY TEST")
    print(f"Running regression {N_RUNS} times with RERANKER=OFF")
    print(f"Dataset: {DATASET}")
    print(f"Deployment: {DEPLOYMENT}")
    print("=" * 60)
    
    rows = load_dataset()
    print(f"Loaded {len(rows)} questions")
    
    # Track results per question across runs
    question_results = {row.get("id"): [] for row in rows}
    run_summaries = []
    
    for run_idx in range(N_RUNS):
        print(f"\n--- Run {run_idx + 1}/{N_RUNS} ---")
        
        results = run_regression_once(rows, run_idx)
        
        passed = sum(1 for r in results.values() if r.get("success"))
        failed = len(results) - passed
        
        run_summaries.append({
            "run": run_idx + 1,
            "passed": passed,
            "failed": failed,
            "pass_rate": passed / len(results) if results else 0,
        })
        
        for qid, result in results.items():
            question_results[qid].append(result.get("success", False))
        
        print(f"  Total: {passed}/{len(results)} passed ({100*passed/len(results):.1f}%)")
        
        # Brief pause between runs
        if run_idx < N_RUNS - 1:
            time.sleep(2)
    
    # Analyze stability
    print("\n" + "=" * 60)
    print("STABILITY ANALYSIS")
    print("=" * 60)
    
    stability_report = []
    for qid, passes in question_results.items():
        pass_count = sum(passes)
        fail_count = N_RUNS - pass_count
        
        if pass_count == N_RUNS:
            status = "STABLE_PASS"
        elif pass_count == 0:
            status = "STABLE_FAIL"
        else:
            status = "FLAKY"
        
        stability_report.append({
            "qid": qid,
            "pass_count": pass_count,
            "fail_count": fail_count,
            "status": status,
            "pass_rate": pass_count / N_RUNS,
        })
        
        print(f"  {qid}: {pass_count}/{N_RUNS} passed ({status})")
    
    # Generate markdown report
    report = f"""# Reranker Regression Stability Test

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}
Runs: {N_RUNS}
Reranker: **OFF**
Deployment: {DEPLOYMENT}

## Summary

| Metric | Value |
|--------|-------|
| Total Runs | {N_RUNS} |
| Questions per Run | {len(rows)} |
| Mean Pass Rate | {sum(r['pass_rate'] for r in run_summaries) / N_RUNS * 100:.1f}% |

## Per-Run Results

| Run | Passed | Failed | Pass Rate |
|-----|--------|--------|-----------|
"""
    
    for r in run_summaries:
        report += f"| {r['run']} | {r['passed']} | {r['failed']} | {r['pass_rate']*100:.1f}% |\n"
    
    report += """
## Per-Question Stability

| Question | Pass Count | Fail Count | Status |
|----------|------------|------------|--------|
"""
    
    for s in stability_report:
        report += f"| {s['qid']} | {s['pass_count']}/{N_RUNS} | {s['fail_count']}/{N_RUNS} | {s['status']} |\n"
    
    # Focus on synth-006
    synth006 = next((s for s in stability_report if s['qid'] == 'synth-006'), None)
    
    report += f"""
## Key Finding: synth-006

"""
    
    if synth006:
        if synth006['status'] == 'STABLE_PASS':
            report += f"""**synth-006 PASSES consistently** ({synth006['pass_count']}/{N_RUNS} runs).

The earlier failure was truly transient (likely cold-start/timing).
Reranker OFF does NOT cause synth-006 failures.
"""
        elif synth006['status'] == 'STABLE_FAIL':
            report += f"""**synth-006 FAILS consistently** ({synth006['fail_count']}/{N_RUNS} runs).

Reranker provides real robustness for this question.
Recommendation: Keep reranker ON for global_synthesis intents.
"""
        else:
            report += f"""**synth-006 is FLAKY** ({synth006['pass_count']}/{N_RUNS} passes, {synth006['fail_count']}/{N_RUNS} fails).

The issue is nondeterminism (DB warmup, timing, caching, or LLM variance).
Reranker may help but is not the root cause.
"""
    
    report += """
## Conclusion

"""
    
    flaky = [s for s in stability_report if s['status'] == 'FLAKY']
    stable_fail = [s for s in stability_report if s['status'] == 'STABLE_FAIL']
    
    if len(flaky) > 0:
        report += f"- **{len(flaky)} flaky questions** detected: {', '.join(s['qid'] for s in flaky)}\n"
        report += "- Root cause is likely nondeterminism, not reranker\n"
    
    if len(stable_fail) > 0:
        report += f"- **{len(stable_fail)} consistently failing questions**: {', '.join(s['qid'] for s in stable_fail)}\n"
    
    if len(flaky) == 0 and len(stable_fail) == 0:
        report += "- All questions pass consistently with reranker OFF\n"
        report += "- Reranker is not required for regression stability\n"
    
    # Write report
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(report, encoding="utf-8")
    print(f"\nReport written to: {REPORT_PATH}")
    
    # Write raw data
    raw_data = {
        "timestamp": datetime.now().isoformat(),
        "n_runs": N_RUNS,
        "reranker": "OFF",
        "run_summaries": run_summaries,
        "stability_report": stability_report,
    }
    (OUT_DIR / "stability_test.json").write_text(
        json.dumps(raw_data, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
