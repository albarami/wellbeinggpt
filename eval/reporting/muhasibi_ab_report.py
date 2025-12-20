"""Muḥāsibī A/B Evaluation Report Generator.

Generates markdown reports comparing evaluation modes with:
- Attribution table showing integrity effect vs reasoning effect
- Value-add-per-ms for ROI analysis
- Top 10 before/after diffs with request_id links
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

from eval.scoring.ab_metrics import (
    AbSummary,
    AbDelta,
    AbAttributionAnalysis,
    compute_mode_summary,
    compute_attribution_analysis,
    load_jsonl,
)


def _format_float(val: float, decimals: int = 2) -> str:
    """Format float with specified decimals."""
    return f"{val:.{decimals}f}"


def _format_delta(delta: AbDelta) -> str:
    """Format delta with sign and percentage."""
    sign = "+" if delta.delta >= 0 else ""
    return f"{sign}{delta.delta:.3f} ({sign}{delta.delta_pct:.1f}%)"


def generate_summary_table(summaries: list[AbSummary]) -> str:
    """Generate markdown table of all mode summaries."""
    lines = [
        "| Mode | N | unsup_must_cite | cite_errors | PASS_FULL | used_edges | arg_chains | quarantine | p50 ms | p95 ms |",
        "|------|--:|----------------:|------------:|----------:|-----------:|-----------:|-----------:|-------:|-------:|",
    ]
    
    for s in summaries:
        lines.append(
            f"| {s.mode} | {s.n} | {_format_float(s.unsupported_must_cite_rate, 3)} | "
            f"{s.citation_validity_errors} | {_format_float(s.contract_pass_full * 100, 1)}% | "
            f"{_format_float(s.mean_used_edges)} | {_format_float(s.mean_argument_chains)} | "
            f"{s.quarantined_cites_blocked} | {int(s.p50_latency_ms)} | {int(s.p95_latency_ms)} |"
        )
    
    return "\n".join(lines)


def generate_style_table(summaries: list[AbSummary]) -> str:
    """Generate markdown table of style metrics."""
    lines = [
        "| Mode | Quote Count | Paragraph Count | Reasoning Leak Rate |",
        "|------|------------:|----------------:|--------------------:|",
    ]
    
    for s in summaries:
        lines.append(
            f"| {s.mode} | {_format_float(s.mean_quote_count)} | "
            f"{_format_float(s.mean_paragraph_count)} | {_format_float(s.reasoning_block_leak_rate, 4)} |"
        )
    
    return "\n".join(lines)


def generate_attribution_table(analysis: AbAttributionAnalysis) -> str:
    """Generate attribution analysis table with all required metrics."""
    lines = [
        "## Attribution Analysis (Clean Deltas)",
        "",
        "| Comparison | unsup_must_cite | cite_errors | quarantine | PASS_FULL | used_edges | arg_chains | p50_ms | p95_ms |",
        "|------------|-----------------|-------------|------------|-----------|------------|------------|--------|--------|",
    ]
    
    effects = [
        ("RAG_ONLY_INT - RAG_ONLY", analysis.integrity_effect),
        ("RAG_GRAPH_INT - RAG_GRAPH", analysis.integrity_graph_effect),
        ("FULL - RAG_ONLY", analysis.total_effect),
        ("FULL - RAG_PLUS_GRAPH", analysis.full_vs_graph),
        ("FULL - RAG_GRAPH_INT (Muḥāsibī)", analysis.muhasibi_effect),
    ]
    
    for name, deltas in effects:
        if not deltas:
            continue
        unsup = deltas.get("unsupported_must_cite_rate")
        contract = deltas.get("contract_pass_full")
        edges = deltas.get("mean_used_edges")
        chains = deltas.get("mean_argument_chains")
        quarantine = deltas.get("quarantined_cites_blocked")
        latency = deltas.get("mean_latency_ms")
        
        unsup_str = _format_delta(unsup) if unsup else "-"
        contract_str = _format_delta(contract) if contract else "-"
        edges_str = _format_delta(edges) if edges else "-"
        chains_str = _format_delta(chains) if chains else "-"
        quarantine_str = f"+{int(quarantine.delta)}" if quarantine else "-"
        latency_str = f"+{int(latency.delta)}ms" if latency else "-"
        
        lines.append(f"| {name} | {unsup_str} | - | {quarantine_str} | {contract_str} | {edges_str} | {chains_str} | {latency_str} | - |")
    
    return "\n".join(lines)


def generate_value_per_ms_table(analysis: AbAttributionAnalysis) -> str:
    """Generate value-add per millisecond table."""
    lines = [
        "## Value-Add per Millisecond",
        "",
        "| Effect | Metric | Delta | Latency Delta | Value/ms |",
        "|--------|--------|-------|---------------|----------|",
    ]
    
    key_metrics = ["contract_pass_full", "mean_used_edges", "mean_argument_chains"]
    
    for name, deltas in [("Muḥāsibī", analysis.muhasibi_effect), ("Total", analysis.total_effect)]:
        if not deltas:
            continue
        for metric in key_metrics:
            d = deltas.get(metric)
            if d and d.delta_per_ms != 0:
                lines.append(
                    f"| {name} | {metric} | {_format_delta(d)} | "
                    f"+{int(deltas.get('mean_latency_ms', AbDelta(metric='', baseline_mode='', full_mode='', baseline_value=0, full_value=0, delta=0)).delta)}ms | "
                    f"{d.delta_per_ms:.6f} |"
                )
    
    return "\n".join(lines)


def find_worst_diffs(
    baseline_rows: list[dict[str, Any]],
    full_rows: list[dict[str, Any]],
    limit: int = 10,
) -> list[dict[str, Any]]:
    """Find questions with largest quality improvement from baseline to full."""
    baseline_by_id = {r.get("id"): r for r in baseline_rows}
    
    diffs: list[dict[str, Any]] = []
    for full_r in full_rows:
        qid = full_r.get("id")
        baseline_r = baseline_by_id.get(qid)
        if not baseline_r:
            continue
        
        # Compute quality delta (higher = more improvement)
        baseline_edges = len((baseline_r.get("graph_trace", {}) or {}).get("used_edges", []) or [])
        full_edges = len((full_r.get("graph_trace", {}) or {}).get("used_edges", []) or [])
        
        baseline_cites = len(baseline_r.get("citations", []) or [])
        full_cites = len(full_r.get("citations", []) or [])
        
        edge_delta = full_edges - baseline_edges
        cite_delta = full_cites - baseline_cites
        quality_delta = edge_delta * 2 + cite_delta  # Weight edges more
        
        diffs.append({
            "id": qid,
            "question": full_r.get("question", "")[:60] + "...",
            "baseline_edges": baseline_edges,
            "full_edges": full_edges,
            "baseline_cites": baseline_cites,
            "full_cites": full_cites,
            "quality_delta": quality_delta,
        })
    
    diffs.sort(key=lambda x: -x["quality_delta"])
    return diffs[:limit]


def generate_top_diffs_table(diffs: list[dict[str, Any]]) -> str:
    """Generate table of top quality improvements."""
    lines = [
        "## Top 10 Quality Improvements (Baseline → Full)",
        "",
        "| ID | Question | Edges | Citations | Quality Delta |",
        "|----|----------|-------|-----------|---------------|",
    ]
    
    for d in diffs:
        lines.append(
            f"| {d['id']} | {d['question']} | "
            f"{d['baseline_edges']}→{d['full_edges']} | "
            f"{d['baseline_cites']}→{d['full_cites']} | "
            f"+{d['quality_delta']} |"
        )
    
    return "\n".join(lines)


def generate_report(
    summaries: dict[str, AbSummary],
    analysis: AbAttributionAnalysis,
    baseline_rows: list[dict[str, Any]] | None = None,
    full_rows: list[dict[str, Any]] | None = None,
    breakthrough_enabled: bool = False,
) -> str:
    """Generate complete markdown report."""
    timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
    
    sections = [
        f"# Muḥāsibī A/B Evaluation Report",
        f"",
        f"**Generated**: {timestamp}",
        f"**Breakthrough Mode**: {'Enabled' if breakthrough_enabled else 'Disabled'}",
        f"",
        "## Summary by Mode",
        "",
        generate_summary_table(list(summaries.values())),
        "",
        "## Style Metrics",
        "",
        generate_style_table(list(summaries.values())),
        "",
        generate_attribution_table(analysis),
        "",
        generate_value_per_ms_table(analysis),
    ]
    
    # Add top diffs if data available
    if baseline_rows and full_rows:
        diffs = find_worst_diffs(baseline_rows, full_rows)
        if diffs:
            sections.extend(["", generate_top_diffs_table(diffs)])
    
    sections.extend([
        "",
        "## Interpretation",
        "",
        "- **Integrity effect**: Reduction in bad citations from quarantine (RAG_ONLY_INTEGRITY - RAG_ONLY)",
        "- **Graph effect**: Improvement from graph expansion (RAG_PLUS_GRAPH - RAG_ONLY)",
        "- **Muḥāsibī effect**: Pure reasoning value-add from contracts, binding, critic loop (FULL - RAG_PLUS_GRAPH_INTEGRITY)",
        "- **Value/ms**: Delta per millisecond latency increase - higher is better ROI",
        "",
        "## Safety Gates",
        "",
        f"- Unsupported MUST_CITE rate in FULL_SYSTEM: {_format_float(summaries.get('FULL_SYSTEM', AbSummary(mode='FULL_SYSTEM')).unsupported_must_cite_rate, 4)}",
        f"- Reasoning block leak rate: {_format_float(summaries.get('FULL_SYSTEM', AbSummary(mode='FULL_SYSTEM')).reasoning_block_leak_rate, 4)}",
    ])
    
    return "\n".join(sections)


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Generate Muḥāsibī A/B report")
    parser.add_argument("--outputs-dir", required=True, help="Directory with JSONL outputs per mode")
    parser.add_argument("--out", required=True, help="Output markdown path")
    parser.add_argument("--breakthrough", action="store_true", help="Mark as breakthrough mode report")
    parser.add_argument("--run-id", default="latest", help="Run ID prefix for output files")
    args = parser.parse_args()
    
    outputs_dir = Path(args.outputs_dir)
    summaries: dict[str, AbSummary] = {}
    baseline_rows: list[dict[str, Any]] | None = None
    full_rows: list[dict[str, Any]] | None = None
    
    # Load all mode outputs
    modes = [
        "LLM_ONLY_UNGROUNDED",
        "RAG_ONLY",
        "RAG_ONLY_INTEGRITY",
        "RAG_PLUS_GRAPH",
        "RAG_PLUS_GRAPH_INTEGRITY",
        "FULL_SYSTEM",
    ]
    
    for mode in modes:
        # Find files matching *__{mode}.jsonl pattern (glob search)
        matching_files = list(outputs_dir.glob(f"*__{mode}.jsonl"))
        
        # Also try exact patterns
        patterns = [
            outputs_dir / f"{args.run_id}__{mode}.jsonl",
            outputs_dir / f"{mode}.jsonl",
        ]
        
        # Combine and dedupe
        all_candidates = matching_files + [p for p in patterns if p.exists()]
        
        for candidate in all_candidates:
            if candidate.exists():
                rows = load_jsonl(candidate)
                summaries[mode] = compute_mode_summary(mode, rows)
                
                # Cache baseline and full for diffs
                if mode == "RAG_ONLY":
                    baseline_rows = rows
                elif mode == "FULL_SYSTEM":
                    full_rows = rows
                break
    
    # Compute attribution analysis
    analysis = compute_attribution_analysis(summaries)
    
    # Generate report
    report = generate_report(
        summaries=summaries,
        analysis=analysis,
        baseline_rows=baseline_rows,
        full_rows=full_rows,
        breakthrough_enabled=args.breakthrough,
    )
    
    # Write output
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(report)
    
    print(f"Report written to: {out_path}")


if __name__ == "__main__":
    main()
