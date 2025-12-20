"""Deterministic eligibility threshold tuner (Mixed).

Objective:
- Reduce false_abstention_rate on Mixed
- Preserve false_answer_rate == 0.00

This tool does NOT change code automatically. It prints safe suggestions.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from eval.datasets.io import read_dataset_jsonl
from eval.datasets.source_loader import load_dotenv_if_present
from eval.io import read_jsonl_rows
from eval.run_meta import build_run_id, sha256_file
from eval.types import EvalOutputRow


@dataclass(frozen=True)
class EligibilityFeatures:
    top1: float
    mean_top10: float
    topk_len: int


def _features(r: EvalOutputRow) -> EligibilityFeatures:
    scores = [float(x.score) for x in (r.retrieval_trace.top_k_chunks or [])]
    top1 = scores[0] if scores else 0.0
    top10 = scores[: min(10, len(scores))]
    mean_top10 = (sum(top10) / len(top10)) if top10 else 0.0
    return EligibilityFeatures(top1=top1, mean_top10=mean_top10, topk_len=len(scores))


def _load_outputs(path: Path) -> list[EvalOutputRow]:
    raw = read_jsonl_rows(path)
    return [EvalOutputRow(**r) for r in raw]


def tune_mixed(*, dataset_path: Path, out_dir: Path, seed: int, prompts_version: str) -> dict[str, Any]:
    sha = sha256_file(dataset_path)
    run_id = build_run_id(dataset_id="wellbeing", dataset_version="v1", dataset_sha256=sha, seed=seed, prompts_version=prompts_version)

    out_path = out_dir / f"{run_id}__FULL_SYSTEM.jsonl"
    if not out_path.exists():
        raise SystemExit(f"Missing outputs: {out_path}")

    ds = read_dataset_jsonl(dataset_path)
    ds_by_id = {r.id: r.model_dump() for r in ds}

    outs = _load_outputs(out_path)
    out_by_id = {o.id: o for o in outs}

    false_abstentions = []
    false_answers = []

    oos_stats = []
    inscope_stats = []

    for rid, d in ds_by_id.items():
        o = out_by_id.get(rid)
        if o is None:
            continue
        expect_abstain = bool(d.get("expect_abstain"))
        f = _features(o)

        if expect_abstain:
            oos_stats.append((rid, f, o.abstained, o.abstain_reason, o.debug))
        else:
            inscope_stats.append((rid, f, o.abstained, o.abstain_reason, o.debug))

        if (not expect_abstain) and o.abstained:
            false_abstentions.append((rid, f, o.abstain_reason, o.debug))
        if expect_abstain and (not o.abstained):
            false_answers.append((rid, f, o.abstain_reason, o.debug))

    # Hard safety: never suggest thresholds that would increase false answers.
    if false_answers:
        # This should be fixed before tuning thresholds.
        return {
            "run_id": run_id,
            "status": "BLOCKED_FALSE_ANSWERS",
            "false_answers": [
                {"id": rid, "top1": f.top1, "mean_top10": f.mean_top10, "debug": dbg}
                for rid, f, _ar, dbg in false_answers[:20]
            ],
        }

    # Candidate threshold suggestions:
    # We only print safe-lowering suggestions based on the *observed* false-abstentions
    # and the OOS distribution.
    oos_top1 = sorted([f.top1 for _rid, f, _a, _ar, _dbg in oos_stats])
    oos_mean = sorted([f.mean_top10 for _rid, f, _a, _ar, _dbg in oos_stats])

    # Conservative floor: keep thresholds above the 95th percentile of OOS retrieval signals.
    def _p(xs: list[float], p: float) -> float:
        if not xs:
            return 0.0
        i = int((len(xs) - 1) * p)
        return float(xs[i])

    safe_top1_floor = _p(oos_top1, 0.95)
    safe_mean_floor = _p(oos_mean, 0.95)

    # To reduce false abstentions, we can lower thresholds down to at most these floors.
    # We report where the false abstentions sit relative to those floors.
    fa = [
        {
            "id": rid,
            "top1": f.top1,
            "mean_top10": f.mean_top10,
            "abstain_reason": ar,
            "eligibility": (dbg or {}).get("eligibility"),
        }
        for rid, f, ar, dbg in false_abstentions
    ]

    return {
        "run_id": run_id,
        "status": "OK",
        "false_answer_count": len(false_answers),
        "false_abstention_count": len(false_abstentions),
        "suggested_thresholds": {
            "top1_min_at_or_below": safe_top1_floor,
            "mean_topk_min_at_or_below": safe_mean_floor,
            "note": "These are conservative floors based on OOS 95th percentile retrieval signals; apply only if false_answer_rate remains 0.00.",
        },
        "false_abstentions": fa[:30],
    }


def main() -> None:
    load_dotenv_if_present()
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="eval/datasets/mixed_oos.jsonl")
    p.add_argument("--out-dir", default="eval/output")
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--prompts-version", default="v1")
    args = p.parse_args()

    res = tune_mixed(
        dataset_path=Path(args.dataset),
        out_dir=Path(args.out_dir),
        seed=int(args.seed),
        prompts_version=str(args.prompts_version),
    )
    print(json.dumps(res, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
