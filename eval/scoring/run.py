"""Run scoring over eval outputs and write summaries."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from apps.api.core.database import get_session

from eval.datasets.io import read_dataset_jsonl
from eval.datasets.source_loader import load_dotenv_if_present
from eval.io import read_jsonl_rows
from eval.types import EvalMode, EvalOutputRow

from eval.run_meta import build_run_id, sha256_file
from eval.scoring.grounding import score_grounding
from eval.scoring.retrieval import score_retrieval
from eval.scoring.graph import score_graph
from eval.scoring.rubric import score_rubric, score_rubric_row
from eval.scoring.grounding import claim_supported
from eval.scoring.policy_audit import score_policy_audit
from eval.scoring.uplift import summarize


def _load_outputs(path: Path) -> list[EvalOutputRow]:
    raw = read_jsonl_rows(path)
    return [EvalOutputRow(**r) for r in raw]


def _dataset_map(path: Path) -> dict[str, dict[str, Any]]:
    rows = read_dataset_jsonl(path)
    return {r.id: r.model_dump() for r in rows}


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted(list({k for r in rows for k in r.keys()}))
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)


async def score_run(*, run_id: str, dataset_path: Path, output_dir: Path) -> dict[str, Any]:
    load_dotenv_if_present()
    dmap = _dataset_map(dataset_path)
    results: dict[str, Any] = {"run_id": run_id, "modes": {}, "uplift": {}}

    async with get_session() as session:
        for mode in [m.value for m in EvalMode]:
            out_path = output_dir / f"{run_id}__{mode}.jsonl"
            if not out_path.exists():
                continue
            outputs = _load_outputs(out_path)

            grounding = await score_grounding(session=session, outputs=outputs, dataset_by_id=dmap)
            retrieval = score_retrieval(outputs=outputs, dataset_by_id=dmap)
            graph = await score_graph(session=session, outputs=outputs, dataset_by_id=dmap)
            rubric = score_rubric(outputs, dmap)
            policy_audit, policy_examples = score_policy_audit(outputs)

            # Hard gate: any policy audit violation fails the run.
            if policy_audit.violations > 0:
                raise RuntimeError(
                    f"Policy audit failed for run_id={run_id} mode={mode}: "
                    f"{policy_audit.violations} violations"
                )

            results["modes"][mode] = {
                "grounding": asdict(grounding),
                "retrieval": asdict(retrieval),
                "graph": asdict(graph),
                "rubric": asdict(rubric),
                "policy_audit": {**asdict(policy_audit), "examples": policy_examples},
            }

        # AlMuhasbi uplift: FULL_SYSTEM vs RAG_PLUS_GRAPH (A/B)
        full_path = output_dir / f"{run_id}__FULL_SYSTEM.jsonl"
        rag_path = output_dir / f"{run_id}__RAG_PLUS_GRAPH.jsonl"
        if full_path.exists() and rag_path.exists():
            full_rows = _load_outputs(full_path)
            rag_rows = _load_outputs(rag_path)
            rag_by_id = {r.id: r for r in rag_rows}

            deltas_unsupported: list[float] = []
            deltas_rubric: list[float] = []
            deltas_cross_hit: list[float] = []

            for fr in full_rows:
                rr = rag_by_id.get(fr.id)
                if rr is None:
                    continue
                d = dmap.get(fr.id, {})

                async def _unsupported_rate(row: EvalOutputRow) -> float:
                    total = 0
                    bad = 0
                    for cl in row.claims:
                        cld = cl.model_dump()
                        if not bool(cld.get("requires_evidence", True)):
                            continue
                        if cld.get("support_policy") in {"no_cite_allowed", "may_cite"}:
                            continue
                        total += 1
                        ok = await claim_supported(session, row, cld)
                        if not ok:
                            bad += 1
                    return (bad / total) if total else 0.0

                full_u = await _unsupported_rate(fr)
                rag_u = await _unsupported_rate(rr)
                # Improvement is reduction in unsupported rate (positive if FULL is better).
                deltas_unsupported.append(rag_u - full_u)

                deltas_rubric.append(float(score_rubric_row(fr, d)) - float(score_rubric_row(rr, d)))

                if str(d.get("type")) == "cross_pillar":
                    full_hit = 1.0 if (fr.graph_trace.edges or fr.graph_trace.paths) else 0.0
                    rag_hit = 1.0 if (rr.graph_trace.edges or rr.graph_trace.paths) else 0.0
                    deltas_cross_hit.append(full_hit - rag_hit)

            results["uplift"] = {
                "unsupported_claim_rate_delta": asdict(summarize(deltas_unsupported)),
                "rubric_score_delta": asdict(summarize(deltas_rubric)),
                "cross_pillar_hit_delta": asdict(summarize(deltas_cross_hit)) if deltas_cross_hit else None,
            }

        # World Model uplift: FULL_SYSTEM_BREAKTHROUGH_WORLD_MODEL vs FULL_SYSTEM (A/B)
        wm_path = output_dir / f"{run_id}__FULL_SYSTEM_BREAKTHROUGH_WORLD_MODEL.jsonl"
        full_path = output_dir / f"{run_id}__FULL_SYSTEM.jsonl"
        if wm_path.exists() and full_path.exists():
            from eval.scoring.world_model import score_world_model_answer
            
            wm_rows = _load_outputs(wm_path)
            full_rows = _load_outputs(full_path)
            full_by_id = {r.id: r for r in full_rows}
            
            deltas_loop_relevance: list[float] = []
            deltas_intervention_completeness: list[float] = []
            deltas_mechanism_coverage: list[float] = []
            deltas_pillar_coverage: list[float] = []
            
            for wr in wm_rows:
                fr = full_by_id.get(wr.id)
                if fr is None:
                    continue
                d = dmap.get(wr.id, {})
                
                # Score world model row
                wm_trace = wr.mechanism_trace.model_dump() if wr.mechanism_trace else {}
                question_entities = d.get("detected_entities", [])
                question_pillars = d.get("detected_pillars", [])
                
                wm_scores = score_world_model_answer(
                    answer_ar=wr.answer_ar,
                    mechanism_trace=wm_trace,
                    question_entities=question_entities,
                    question_pillars=question_pillars,
                )
                
                # Full system baseline (no mechanism trace)
                full_scores = score_world_model_answer(
                    answer_ar=fr.answer_ar,
                    mechanism_trace={},
                    question_entities=question_entities,
                    question_pillars=question_pillars,
                )
                
                # Compute deltas (world model - full system)
                deltas_loop_relevance.append(
                    wm_scores["loop_relevance"] - full_scores["loop_relevance"]
                )
                deltas_intervention_completeness.append(
                    wm_scores["intervention_completeness"] - full_scores["intervention_completeness"]
                )
                deltas_mechanism_coverage.append(
                    wm_scores["mechanism_coverage"] - full_scores["mechanism_coverage"]
                )
                deltas_pillar_coverage.append(
                    wm_scores["pillar_coverage"] - full_scores["pillar_coverage"]
                )
            
            results["world_model_uplift"] = {
                "loop_relevance_delta": asdict(summarize(deltas_loop_relevance)),
                "intervention_completeness_delta": asdict(summarize(deltas_intervention_completeness)),
                "mechanism_coverage_delta": asdict(summarize(deltas_mechanism_coverage)),
                "pillar_coverage_delta": asdict(summarize(deltas_pillar_coverage)),
            }

    # Write summary json
    summary_path = output_dir / f"{run_id}__summary.json"
    summary_path.write_text(json.dumps(results, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")

    # Write a flat CSV
    csv_rows: list[dict[str, Any]] = []
    for mode, mres in results["modes"].items():
        flat = {"run_id": run_id, "mode": mode}
        for section, vals in mres.items():
            for k, v in vals.items():
                flat[f"{section}.{k}"] = v
        csv_rows.append(flat)

    _write_csv(output_dir / f"{run_id}__summary.csv", csv_rows)
    return results


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--output-dir", default="eval/output")
    p.add_argument("--dataset-id", default="wellbeing")
    p.add_argument("--dataset-version", default="v1")
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--prompts-version", default="v1")
    p.add_argument("--run-id", default=None, help="Optional; computed from dataset hash if omitted.")
    args = p.parse_args()

    import asyncio

    dataset_path = Path(args.dataset)
    run_id = str(args.run_id or "").strip()
    if not run_id:
        sha = sha256_file(dataset_path)
        run_id = build_run_id(
            dataset_id=str(args.dataset_id),
            dataset_version=str(args.dataset_version),
            dataset_sha256=sha,
            seed=int(args.seed),
            prompts_version=str(args.prompts_version),
        )

    asyncio.run(
        score_run(
            run_id=run_id,
            dataset_path=dataset_path,
            output_dir=Path(args.output_dir),
        )
    )


if __name__ == "__main__":
    main()
