"""Deterministic evaluation runner CLI.

Reason: keep `eval/runner.py` <500 LOC (project rule).
"""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from eval.runner_core import RunnerConfig, run_dataset


def _cli() -> RunnerConfig:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, help="Path to dataset JSONL")
    p.add_argument("--dataset-id", default="wellbeing")
    p.add_argument("--dataset-version", default="v1")
    p.add_argument("--out-dir", default="eval/output")
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--prompts-version", default="v1")
    p.add_argument("--no-llm-only", action="store_true")
    p.add_argument("--start", type=int, default=0, help="Start offset within dataset rows (0-based)")
    p.add_argument("--limit", type=int, default=None, help="Limit number of dataset rows")
    args = p.parse_args()

    return RunnerConfig(
        dataset_path=Path(args.dataset),
        dataset_id=str(args.dataset_id),
        dataset_version=str(args.dataset_version),
        out_dir=Path(args.out_dir),
        seed=int(args.seed),
        top_k=int(args.top_k),
        prompts_version=str(args.prompts_version),
        include_llm_only=not bool(args.no_llm_only),
        start=int(args.start or 0),
        limit=args.limit,
    )


def main() -> None:
    cfg = _cli()
    run_id = asyncio.run(run_dataset(cfg))
    print(run_id)


if __name__ == "__main__":
    main()
