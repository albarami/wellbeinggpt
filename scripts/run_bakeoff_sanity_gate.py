"""Bakeoff Sanity Gate - Production Release Check

A 50-question subset for nightly/weekly scheduled runs.
Not for every PR - just to prevent silent drift.

Thresholds (conservative):
- PASS_FULL rate >= 95%
- Unexpected fail rate <= 3%
- Citation present rate >= 95%
- Mean used edges >= baseline (for cross-pillar)

Usage:
    python scripts/run_bakeoff_sanity_gate.py

Exit codes:
    0 = Gate passed
    1 = Gate failed
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any

# Fix Windows console encoding
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import requests
from dotenv import load_dotenv

load_dotenv()

# Configuration
API_BASE = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")
DATASET_PATH = Path("eval/datasets/bakeoff_sanity_gate_v1.jsonl")
OUTPUT_DIR = Path("eval/output/sanity_gate")
REPORT_PATH = Path("eval/reports/bakeoff_sanity_gate.md")

# Thresholds
THRESHOLDS = {
    "pass_full_rate": 0.95,
    "unexpected_fail_rate_max": 0.03,
    "citation_present_rate": 0.95,
    "mean_used_edges_min": 2.0,
}


def compute_dataset_hash(path: Path) -> str:
    """SHA256 of dataset for reproducibility."""
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()[:16]


def run_question(q: dict[str, Any]) -> dict[str, Any]:
    """Run a single question through the API."""
    mode = "natural_chat" if q.get("type") == "natural_chat" else "answer"
    payload = {
        "question": q.get("question"),
        "mode": mode,
        "lang": "ar",
        "engine": "muhasibi",
    }
    
    t0 = time.perf_counter()
    try:
        response = requests.post(f"{API_BASE}/ask/ui", json=payload, timeout=120)
        latency_ms = int((time.perf_counter() - t0) * 1000)
        
        if response.status_code != 200:
            return {
                "qid": q.get("id"),
                "status": "ERROR",
                "error": f"HTTP {response.status_code}",
                "latency_ms": latency_ms,
            }
        
        data = response.json()
        return {
            "qid": q.get("id"),
            "type": q.get("type"),
            "contract": data.get("contract_outcome"),
            "citations": len(data.get("citations_spans", [])),
            "used_edges": len(data.get("graph_trace", {}).get("used_edges", [])),
            "abstained": bool(data.get("abstained") or data.get("abstain_reason")),
            "expect_abstain": q.get("expect_abstain", False),
            "latency_ms": latency_ms,
            "status": "OK",
        }
    except Exception as e:
        return {
            "qid": q.get("id"),
            "status": "ERROR",
            "error": str(e)[:100],
            "latency_ms": int((time.perf_counter() - t0) * 1000),
        }


def compute_metrics(results: list[dict[str, Any]]) -> dict[str, float]:
    """Compute gate metrics from results."""
    total = len(results)
    if total == 0:
        return {}
    
    pass_full_count = sum(1 for r in results if r.get("contract") == "PASS_FULL")
    citation_present_count = sum(1 for r in results if r.get("citations", 0) > 0)
    
    # Unexpected failures: abstained when shouldn't OR error
    unexpected_fails = sum(
        1 for r in results
        if (r.get("status") == "ERROR") or
           (r.get("abstained") and not r.get("expect_abstain"))
    )
    
    # Mean used edges for cross_pillar questions
    cross_pillar_edges = [
        r.get("used_edges", 0)
        for r in results
        if r.get("type") == "cross_pillar" and r.get("status") == "OK"
    ]
    mean_edges = mean(cross_pillar_edges) if cross_pillar_edges else 0.0
    
    return {
        "pass_full_rate": pass_full_count / total,
        "citation_present_rate": citation_present_count / total,
        "unexpected_fail_rate": unexpected_fails / total,
        "mean_used_edges": mean_edges,
        "total_questions": total,
        "pass_full_count": pass_full_count,
        "unexpected_fail_count": unexpected_fails,
    }


def check_thresholds(metrics: dict[str, float]) -> tuple[bool, list[str]]:
    """Check if metrics meet thresholds."""
    failures = []
    
    if metrics.get("pass_full_rate", 0) < THRESHOLDS["pass_full_rate"]:
        failures.append(
            f"PASS_FULL rate {metrics['pass_full_rate']:.1%} < {THRESHOLDS['pass_full_rate']:.0%}"
        )
    
    if metrics.get("unexpected_fail_rate", 1) > THRESHOLDS["unexpected_fail_rate_max"]:
        failures.append(
            f"Unexpected fail rate {metrics['unexpected_fail_rate']:.1%} > {THRESHOLDS['unexpected_fail_rate_max']:.0%}"
        )
    
    if metrics.get("citation_present_rate", 0) < THRESHOLDS["citation_present_rate"]:
        failures.append(
            f"Citation rate {metrics['citation_present_rate']:.1%} < {THRESHOLDS['citation_present_rate']:.0%}"
        )
    
    if metrics.get("mean_used_edges", 0) < THRESHOLDS["mean_used_edges_min"]:
        failures.append(
            f"Mean used edges {metrics['mean_used_edges']:.1f} < {THRESHOLDS['mean_used_edges_min']:.1f}"
        )
    
    return len(failures) == 0, failures


def write_report(
    metrics: dict[str, float],
    passed: bool,
    failures: list[str],
    dataset_hash: str,
    results: list[dict[str, Any]],
) -> None:
    """Write markdown report."""
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    status_emoji = "✅" if passed else "❌"
    
    lines = [
        "## Bakeoff Sanity Gate Report",
        "",
        f"- **Generated**: {datetime.utcnow().isoformat()}Z",
        f"- **Status**: {status_emoji} {'PASSED' if passed else 'FAILED'}",
        f"- **Dataset**: `eval/datasets/bakeoff_sanity_gate_v1.jsonl`",
        f"- **Dataset hash**: `{dataset_hash}`",
        f"- **Total questions**: {metrics.get('total_questions', 0)}",
        "",
        "### Metrics",
        "",
        "| Metric | Value | Threshold | Status |",
        "|--------|-------|-----------|--------|",
        f"| PASS_FULL rate | {metrics.get('pass_full_rate', 0):.1%} | >= {THRESHOLDS['pass_full_rate']:.0%} | {'OK' if metrics.get('pass_full_rate', 0) >= THRESHOLDS['pass_full_rate'] else 'FAIL'} |",
        f"| Unexpected fail rate | {metrics.get('unexpected_fail_rate', 0):.1%} | <= {THRESHOLDS['unexpected_fail_rate_max']:.0%} | {'OK' if metrics.get('unexpected_fail_rate', 1) <= THRESHOLDS['unexpected_fail_rate_max'] else 'FAIL'} |",
        f"| Citation present rate | {metrics.get('citation_present_rate', 0):.1%} | >= {THRESHOLDS['citation_present_rate']:.0%} | {'OK' if metrics.get('citation_present_rate', 0) >= THRESHOLDS['citation_present_rate'] else 'FAIL'} |",
        f"| Mean used edges (cross-pillar) | {metrics.get('mean_used_edges', 0):.1f} | >= {THRESHOLDS['mean_used_edges_min']:.1f} | {'OK' if metrics.get('mean_used_edges', 0) >= THRESHOLDS['mean_used_edges_min'] else 'FAIL'} |",
        "",
    ]
    
    if failures:
        lines.append("### Threshold Failures")
        lines.append("")
        for f in failures:
            lines.append(f"- {f}")
        lines.append("")
    
    # List any unexpected failures
    unexpected = [r for r in results if r.get("status") == "ERROR" or (r.get("abstained") and not r.get("expect_abstain"))]
    if unexpected:
        lines.append("### Unexpected Failures")
        lines.append("")
        lines.append("| Question ID | Type | Status | Reason |")
        lines.append("|-------------|------|--------|--------|")
        for r in unexpected:
            reason = r.get("error", "Abstained unexpectedly")
            lines.append(f"| {r.get('qid')} | {r.get('type', 'N/A')} | {r.get('status')} | {reason} |")
        lines.append("")
    
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    
    print(f"Report written to: {REPORT_PATH}")


def main() -> int:
    """Run the sanity gate."""
    print("=" * 60)
    print("BAKEOFF SANITY GATE")
    print("=" * 60)
    
    # Load dataset
    if not DATASET_PATH.exists():
        print(f"ERROR: Dataset not found: {DATASET_PATH}")
        return 1
    
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        questions = [json.loads(line) for line in f]
    
    dataset_hash = compute_dataset_hash(DATASET_PATH)
    print(f"Dataset: {DATASET_PATH}")
    print(f"Hash: {dataset_hash}")
    print(f"Questions: {len(questions)}")
    print()
    
    # Run questions
    results = []
    for i, q in enumerate(questions):
        qid = q.get("id", f"q{i}")
        print(f"[{i+1}/{len(questions)}] {qid}... ", end="", flush=True)
        result = run_question(q)
        results.append(result)
        
        if result["status"] == "OK":
            contract = result.get("contract", "N/A")
            cites = result.get("citations", 0)
            edges = result.get("used_edges", 0)
            print(f"{contract} (cites={cites}, edges={edges})")
        else:
            print(f"ERROR: {result.get('error', 'Unknown')}")
    
    # Save results
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / f"sanity_gate_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.jsonl"
    with open(output_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\nResults saved to: {output_path}")
    
    # Compute metrics
    metrics = compute_metrics(results)
    print()
    print("METRICS:")
    print(f"  PASS_FULL rate: {metrics.get('pass_full_rate', 0):.1%}")
    print(f"  Citation present rate: {metrics.get('citation_present_rate', 0):.1%}")
    print(f"  Unexpected fail rate: {metrics.get('unexpected_fail_rate', 0):.1%}")
    print(f"  Mean used edges (cross-pillar): {metrics.get('mean_used_edges', 0):.1f}")
    
    # Check thresholds
    passed, failures = check_thresholds(metrics)
    
    # Write report
    write_report(metrics, passed, failures, dataset_hash, results)
    
    print()
    if passed:
        print("=" * 60)
        print("SANITY GATE: PASSED")
        print("=" * 60)
        return 0
    else:
        print("=" * 60)
        print("SANITY GATE: FAILED")
        for f in failures:
            print(f"  - {f}")
        print("=" * 60)
        return 1


if __name__ == "__main__":
    sys.exit(main())
