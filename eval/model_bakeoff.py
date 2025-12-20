"""Model Bakeoff CLI: Compare GPT-5 variants across 6 dimensions.

Usage:
    python -m eval.model_bakeoff --models gpt-5-chat,gpt-5.1,gpt-5.2 \
        --datasets stakeholder_v2_hard,world_model_synthesis \
        --modes answer,natural_chat

Outputs:
    eval/reports/model_bakeoff.md
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from eval.model_bakeoff_metrics import (
    BakeoffMetrics,
    compute_all_metrics,
    DimensionScores,
    compute_weighted_score,
)
from eval.model_bakeoff_runner import run_model_on_dataset
from eval.datasets.source_loader import load_dotenv_if_present


@dataclass
class BakeoffConfig:
    """Configuration for model bakeoff."""

    models: list[str]
    datasets: list[str]
    modes: list[str]
    out_dir: Path = Path("eval/reports")
    seed: int = 1337
    temperature: float = 0.1
    max_tokens: int = 2000
    top_k: int = 10
    timeout_per_question: int = 120


@dataclass
class ModelDatasetResult:
    """Results for one model on one dataset/mode."""

    model: str
    dataset: str
    mode: str
    metrics: BakeoffMetrics
    dimension_scores: DimensionScores
    weighted_score: float
    disqualified: bool = False
    disqualify_reason: str = ""


@dataclass
class BakeoffReport:
    """Complete bakeoff report."""

    config: BakeoffConfig
    results: list[ModelDatasetResult] = field(default_factory=list)
    generated_at: str = ""

    def model_summary(self) -> dict[str, dict[str, float]]:
        """Aggregate scores per model across all datasets/modes."""
        by_model: dict[str, list[ModelDatasetResult]] = {}
        for r in self.results:
            by_model.setdefault(r.model, []).append(r)

        out: dict[str, dict[str, float]] = {}
        for m, rs in by_model.items():
            valid = [r for r in rs if not r.disqualified]
            if not valid:
                out[m] = {"weighted_score": -999, "disqualified": 1}
                continue

            avg_weighted = sum(r.weighted_score for r in valid) / len(valid)
            avg_depth = sum(r.dimension_scores.depth for r in valid) / len(valid)
            avg_conn = sum(r.dimension_scores.connections for r in valid) / len(valid)
            avg_nat = sum(r.dimension_scores.naturalness for r in valid) / len(valid)
            avg_speed = sum(r.dimension_scores.speed for r in valid) / len(valid)
            any_disq = any(r.disqualified for r in rs)

            out[m] = {
                "weighted_score": round(avg_weighted, 2),
                "depth": round(avg_depth, 2),
                "connections": round(avg_conn, 2),
                "naturalness": round(avg_nat, 2),
                "speed": round(avg_speed, 2),
                "disqualified": 1 if any_disq else 0,
            }
        return out


def _resolve_dataset_path(name: str) -> Path:
    """Resolve dataset name to path."""
    candidates = [
        Path(f"eval/datasets/{name}.jsonl"),
        Path(f"eval/datasets/{name}"),
        Path(name),
    ]
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError(f"Dataset not found: {name}")


async def run_bakeoff(cfg: BakeoffConfig) -> BakeoffReport:
    """Run the full bakeoff comparison."""
    load_dotenv_if_present()

    report = BakeoffReport(
        config=cfg,
        generated_at=datetime.now(timezone.utc).isoformat(),
    )

    total = len(cfg.models) * len(cfg.datasets) * len(cfg.modes)
    done = 0

    for model in cfg.models:
        for dataset_name in cfg.datasets:
            dataset_path = _resolve_dataset_path(dataset_name)

            for mode in cfg.modes:
                done += 1
                print(f"[{done}/{total}] {model} / {dataset_name} / {mode}")

                try:
                    metrics = await run_model_on_dataset(
                        model=model,
                        dataset_path=dataset_path,
                        mode=mode,
                        seed=cfg.seed,
                        temperature=cfg.temperature,
                        max_tokens=cfg.max_tokens,
                        top_k=cfg.top_k,
                        timeout=cfg.timeout_per_question,
                    )

                    dim_scores = compute_weighted_score(metrics)

                    # Hard disqualification rules
                    disqualified = False
                    reason = ""
                    if metrics.citation_validity_errors > 0:
                        disqualified = True
                        reason = f"citation_validity_errors={metrics.citation_validity_errors}"
                    elif metrics.unsupported_must_cite_rate > 0.05:
                        disqualified = True
                        reason = f"unsupported_must_cite_rate={metrics.unsupported_must_cite_rate:.2%}"

                    result = ModelDatasetResult(
                        model=model,
                        dataset=dataset_name,
                        mode=mode,
                        metrics=metrics,
                        dimension_scores=dim_scores,
                        weighted_score=dim_scores.weighted_total,
                        disqualified=disqualified,
                        disqualify_reason=reason,
                    )
                    report.results.append(result)

                except Exception as e:
                    print(f"  ERROR: {e}")
                    # Record as failed
                    empty_metrics = BakeoffMetrics()
                    empty_dims = DimensionScores()
                    report.results.append(
                        ModelDatasetResult(
                            model=model,
                            dataset=dataset_name,
                            mode=mode,
                            metrics=empty_metrics,
                            dimension_scores=empty_dims,
                            weighted_score=-999,
                            disqualified=True,
                            disqualify_reason=str(e)[:100],
                        )
                    )

    return report


def generate_markdown_report(report: BakeoffReport, out_path: Path) -> None:
    """Generate markdown report."""
    lines: list[str] = []
    lines.append("# Model Bakeoff Report")
    lines.append("")
    lines.append(f"Generated: {report.generated_at}")
    lines.append(f"Models: {', '.join(report.config.models)}")
    lines.append(f"Datasets: {', '.join(report.config.datasets)}")
    lines.append(f"Modes: {', '.join(report.config.modes)}")
    lines.append("")

    # Summary table
    lines.append("## Summary (Aggregated Across All Datasets/Modes)")
    lines.append("")
    summary = report.model_summary()
    ranked = sorted(summary.items(), key=lambda x: -x[1].get("weighted_score", -999))

    lines.append("| Rank | Model | Weighted Score | Depth | Connections | Naturalness | Speed | Status |")
    lines.append("|------|-------|----------------|-------|-------------|-------------|-------|--------|")
    for i, (model, scores) in enumerate(ranked, 1):
        status = "DISQUALIFIED" if scores.get("disqualified") else "OK"
        lines.append(
            f"| {i} | {model} | {scores.get('weighted_score', -999):.1f} | "
            f"{scores.get('depth', 0):.1f} | {scores.get('connections', 0):.1f} | "
            f"{scores.get('naturalness', 0):.1f} | {scores.get('speed', 0):.1f} | {status} |"
        )
    lines.append("")

    # Winner per dimension
    lines.append("## Winner Per Dimension")
    lines.append("")
    dimensions = ["depth", "connections", "naturalness", "speed", "weighted_score"]
    for dim in dimensions:
        valid = [(m, s) for m, s in summary.items() if not s.get("disqualified")]
        if not valid:
            lines.append(f"- **{dim}**: No valid models")
            continue
        best = max(valid, key=lambda x: x[1].get(dim, -999))
        lines.append(f"- **{dim}**: {best[0]} ({best[1].get(dim, 0):.1f})")
    lines.append("")

    # Detailed per-dataset tables
    lines.append("## Detailed Results Per Dataset")
    lines.append("")

    for dataset in report.config.datasets:
        lines.append(f"### {dataset}")
        lines.append("")

        for mode in report.config.modes:
            lines.append(f"#### Mode: {mode}")
            lines.append("")

            ds_results = [r for r in report.results if r.dataset == dataset and r.mode == mode]
            ds_results.sort(key=lambda x: -x.weighted_score)

            lines.append("| Model | Score | Rubric | Citations | Edges | Pillars | Latency p50 | Status |")
            lines.append("|-------|-------|--------|-----------|-------|---------|-------------|--------|")

            for r in ds_results:
                m = r.metrics
                status = f"DQ: {r.disqualify_reason[:30]}" if r.disqualified else "OK"
                lines.append(
                    f"| {r.model} | {r.weighted_score:.1f} | {m.rubric_score:.1f} | "
                    f"{m.total_citations} | {m.used_edges_count} | {m.distinct_pillars} | "
                    f"{m.latency_p50_ms:.0f}ms | {status} |"
                )
            lines.append("")

    # Raw metrics dump
    lines.append("## Raw Metrics")
    lines.append("")
    lines.append("```json")
    raw = []
    for r in report.results:
        raw.append({
            "model": r.model,
            "dataset": r.dataset,
            "mode": r.mode,
            "weighted_score": r.weighted_score,
            "disqualified": r.disqualified,
            "metrics": {
                "rubric_score": r.metrics.rubric_score,
                "citation_validity_errors": r.metrics.citation_validity_errors,
                "unsupported_must_cite_rate": r.metrics.unsupported_must_cite_rate,
                "used_edges_count": r.metrics.used_edges_count,
                "distinct_pillars": r.metrics.distinct_pillars,
                "pass_full_rate": r.metrics.pass_full_rate,
                "latency_p50_ms": r.metrics.latency_p50_ms,
                "latency_p95_ms": r.metrics.latency_p95_ms,
            },
        })
    lines.append(json.dumps(raw, indent=2, ensure_ascii=False))
    lines.append("```")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nReport written to: {out_path}")


def _cli() -> BakeoffConfig:
    p = argparse.ArgumentParser(description="Model Bakeoff: Compare GPT-5 variants")
    p.add_argument(
        "--models",
        default="gpt-5-chat,gpt-5.1,gpt-5.2",
        help="Comma-separated model deployment names",
    )
    p.add_argument(
        "--datasets",
        default="stakeholder_acceptance_v2_hard,world_model_synthesis",
        help="Comma-separated dataset names",
    )
    p.add_argument(
        "--modes",
        default="answer,natural_chat",
        help="Comma-separated modes",
    )
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--temperature", type=float, default=0.1)
    p.add_argument("--max-tokens", type=int, default=2000)
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--timeout", type=int, default=120, help="Timeout per question (seconds)")
    p.add_argument("--out-dir", default="eval/reports")

    args = p.parse_args()

    return BakeoffConfig(
        models=[m.strip() for m in args.models.split(",") if m.strip()],
        datasets=[d.strip() for d in args.datasets.split(",") if d.strip()],
        modes=[m.strip() for m in args.modes.split(",") if m.strip()],
        seed=args.seed,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        top_k=args.top_k,
        timeout_per_question=args.timeout,
        out_dir=Path(args.out_dir),
    )


def main() -> None:
    cfg = _cli()
    print(f"Starting Model Bakeoff")
    print(f"  Models: {cfg.models}")
    print(f"  Datasets: {cfg.datasets}")
    print(f"  Modes: {cfg.modes}")
    print()

    report = asyncio.run(run_bakeoff(cfg))
    out_path = cfg.out_dir / "model_bakeoff.md"
    generate_markdown_report(report, out_path)

    # Print final ranking
    print("\n" + "=" * 60)
    print("FINAL RANKING")
    print("=" * 60)
    summary = report.model_summary()
    ranked = sorted(summary.items(), key=lambda x: -x[1].get("weighted_score", -999))
    for i, (model, scores) in enumerate(ranked, 1):
        status = " (DISQUALIFIED)" if scores.get("disqualified") else ""
        print(f"  #{i} {model}: {scores.get('weighted_score', -999):.1f}{status}")

    if ranked:
        winner = ranked[0]
        if not winner[1].get("disqualified"):
            print(f"\nWINNER: {winner[0]}")


if __name__ == "__main__":
    main()
