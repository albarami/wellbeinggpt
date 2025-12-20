"""Full 160-question bakeoff for gpt-5.1 only (baseline freeze)."""

from __future__ import annotations

import hashlib
import io
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from statistics import mean, median
from typing import Any

# Note: For Windows encoding, run with: python -X utf8 script.py

import requests
from dotenv import load_dotenv

load_dotenv()

# Configuration
API_BASE = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")
DATASET_PATH = Path("eval/datasets/bakeoff_depth_v1.jsonl")
OUTPUT_DIR = Path("eval/output/baseline_freeze")
REPORT_PATH = Path("eval/reports/baseline_freeze_gpt51.md")


def run_bakeoff():
    """Run full bakeoff for gpt-5.1."""
    print("=" * 60)
    print("FULL BAKEOFF BASELINE FREEZE (gpt-5.1)")
    print("=" * 60)
    
    # Load dataset
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        questions = [json.loads(line) for line in f]
    
    dataset_hash = hashlib.sha256(DATASET_PATH.read_bytes()).hexdigest()[:16]
    print(f"Dataset: {DATASET_PATH}")
    print(f"Hash: {dataset_hash}")
    print(f"Questions: {len(questions)}")
    print()
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_jsonl = OUTPUT_DIR / f"gpt-5.1_bakeoff_{datetime.utcnow().strftime('%Y%m%d')}.jsonl"
    
    results = []
    
    for i, q in enumerate(questions):
        qid = q.get("id", f"q{i}")
        qtext = q.get("question", "")
        qtype = q.get("type", "answer")
        mode = "natural_chat" if qtype == "natural_chat" else "answer"
        expect_abstain = q.get("expect_abstain", False)
        
        payload = {
            "question": qtext,
            "mode": mode,
            "lang": "ar",
            "engine": "muhasibi",
        }
        
        t0 = time.perf_counter()
        try:
            response = requests.post(f"{API_BASE}/ask/ui", json=payload, timeout=180)
            latency_ms = int((time.perf_counter() - t0) * 1000)
            
            if response.status_code != 200:
                result = {
                    "qid": qid,
                    "type": qtype,
                    "status": "ERROR",
                    "error": f"HTTP {response.status_code}",
                    "latency_ms": latency_ms,
                }
            else:
                data = response.json()
                contract = data.get("contract_outcome", "N/A")
                citations = len(data.get("citations_spans", []))
                used_edges = len(data.get("graph_trace", {}).get("used_edges", []))
                abstained = bool(data.get("abstained") or data.get("abstain_reason"))
                
                result = {
                    "qid": qid,
                    "type": qtype,
                    "contract": contract,
                    "citations": citations,
                    "used_edges": used_edges,
                    "abstained": abstained,
                    "expect_abstain": expect_abstain,
                    "latency_ms": latency_ms,
                    "status": "OK",
                }
        except Exception as e:
            result = {
                "qid": qid,
                "type": qtype,
                "status": "ERROR",
                "error": str(e)[:100],
                "latency_ms": int((time.perf_counter() - t0) * 1000),
            }
        
        results.append(result)
        
        # Print progress
        if result["status"] == "OK":
            contract = result.get("contract", "N/A")
            cites = result.get("citations", 0)
            edges = result.get("used_edges", 0)
            print(f"[{i+1}/{len(questions)}] {qid}: {contract} (cites={cites}, edges={edges})")
        else:
            print(f"[{i+1}/{len(questions)}] {qid}: ERROR - {result.get('error', 'Unknown')}")
    
    # Save results
    with open(output_jsonl, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\nResults saved to: {output_jsonl}")
    
    # Compute metrics
    total = len(results)
    ok_results = [r for r in results if r.get("status") == "OK"]
    
    pass_full_count = sum(1 for r in ok_results if r.get("contract") == "PASS_FULL")
    citation_present = sum(1 for r in ok_results if r.get("citations", 0) > 0)
    unexpected_fails = sum(
        1 for r in results
        if r.get("status") == "ERROR" or (r.get("abstained") and not r.get("expect_abstain"))
    )
    
    cross_pillar_edges = [r.get("used_edges", 0) for r in ok_results if r.get("type") == "cross_pillar"]
    mean_edges = mean(cross_pillar_edges) if cross_pillar_edges else 0.0
    
    latencies = [r.get("latency_ms", 0) for r in ok_results]
    
    metrics = {
        "total_questions": total,
        "pass_full_count": pass_full_count,
        "pass_full_rate": pass_full_count / total if total else 0,
        "citation_present_rate": citation_present / total if total else 0,
        "unexpected_fail_count": unexpected_fails,
        "unexpected_fail_rate": unexpected_fails / total if total else 0,
        "mean_used_edges_cross_pillar": mean_edges,
        "median_latency_ms": median(latencies) if latencies else 0,
    }
    
    # Write report
    lines = [
        "## Baseline Freeze: gpt-5.1 Full Bakeoff",
        "",
        f"- **Generated**: {datetime.utcnow().isoformat()}Z",
        f"- **Dataset**: `{DATASET_PATH}`",
        f"- **Dataset hash**: `{dataset_hash}`",
        f"- **Total questions**: {total}",
        f"- **Model**: gpt-5.1 (default)",
        "",
        "### Baseline Metrics (Freeze These)",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| PASS_FULL rate | {metrics['pass_full_rate']:.1%} |",
        f"| Citation present rate | {metrics['citation_present_rate']:.1%} |",
        f"| Unexpected fail rate | {metrics['unexpected_fail_rate']:.1%} |",
        f"| Mean used edges (cross-pillar) | {metrics['mean_used_edges_cross_pillar']:.1f} |",
        f"| Median latency (ms) | {metrics['median_latency_ms']:.0f} |",
        "",
        "### By Question Type",
        "",
        "| Type | Count | PASS_FULL | Rate |",
        "|------|-------|-----------|------|",
    ]
    
    by_type = {}
    for r in ok_results:
        t = r.get("type", "unknown")
        if t not in by_type:
            by_type[t] = {"total": 0, "pass_full": 0}
        by_type[t]["total"] += 1
        if r.get("contract") == "PASS_FULL":
            by_type[t]["pass_full"] += 1
    
    for t, counts in sorted(by_type.items()):
        rate = counts["pass_full"] / counts["total"] if counts["total"] else 0
        lines.append(f"| {t} | {counts['total']} | {counts['pass_full']} | {rate:.1%} |")
    
    lines.append("")
    lines.append("### Unexpected Failures")
    lines.append("")
    
    unexpected = [
        r for r in results
        if r.get("status") == "ERROR" or (r.get("abstained") and not r.get("expect_abstain"))
    ]
    if unexpected:
        lines.append("| QID | Type | Status | Reason |")
        lines.append("|-----|------|--------|--------|")
        for r in unexpected:
            reason = r.get("error", "Abstained unexpectedly")
            lines.append(f"| {r.get('qid')} | {r.get('type', 'N/A')} | {r.get('status')} | {reason} |")
    else:
        lines.append("*None*")
    
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Report written to: {REPORT_PATH}")
    
    # Summary
    print()
    print("=" * 60)
    print("BASELINE FREEZE COMPLETE")
    print("=" * 60)
    print(f"PASS_FULL rate: {metrics['pass_full_rate']:.1%}")
    print(f"Citation present rate: {metrics['citation_present_rate']:.1%}")
    print(f"Unexpected fail rate: {metrics['unexpected_fail_rate']:.1%}")
    print(f"Mean used edges (cross-pillar): {metrics['mean_used_edges_cross_pillar']:.1f}")


if __name__ == "__main__":
    run_bakeoff()
