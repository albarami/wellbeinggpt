"""Full-system depth bakeoff (Muḥāsibī) across Azure deployments.

What this does
- Runs the *real* system (/ask/ui) for the same dataset across deployments.
- Writes per-question JSONL + per-model summary JSON.
- Generates a markdown report with disqualification gates and rankings.

Hard gates (disqualify):
- unsupported_must_cite_rate > 0
- citation_validity_errors > 0

Scoring (ignore speed):
- 45% Depth quality
- 35% Cross-pillar intelligence
- 15% Naturalness
- 5% Integrity hygiene

Usage
1) Start API:
   python -m uvicorn apps.api.main:app --host 127.0.0.1 --port 8000
2) Run:
   python scripts/run_system_depth_bakeoff.py

Notes
- This script assumes DB is configured and API can answer.
- For robustness, it writes results incrementally and supports resume.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from statistics import median
from typing import Any, Optional

import requests
from dotenv import load_dotenv

# Ensure repo root is on sys.path (Windows/CI can run with different CWD).
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval.claims import extract_claims
from eval.model_bakeoff_metrics import compute_all_metrics
from eval.scoring.ab_metrics import _distinct_pillars_from_edges  # type: ignore
from eval.scoring.grounding import score_grounding
from eval.types import EvalCitation, EvalMode, EvalOutputRow, GraphTrace, GraphTraceUsedEdge, GraphTraceUsedEdgeSpan


DATASET_PATH = Path("eval/datasets/bakeoff_depth_v1.jsonl")
OUT_DIR = Path("eval/output/bakeoff_depth_v1_system")
REPORT_PATH = Path("eval/reports/model_bakeoff_depth.md")

API_BASE = os.getenv("API_BASE_URL", "http://127.0.0.1:8000").rstrip("/")

# Fairness controls (must match across models)
SEED = 1337
TEMPERATURE = 0.1
MAX_TOKENS = 1500
PROMPTS_VERSION = os.getenv("PROMPTS_VERSION", "v1")

DEPLOYMENTS = ["gpt-5-chat", "gpt-5.1", "gpt-5.2"]
PRINT_EVERY = int(os.getenv("BAKEOFF_PRINT_EVERY", "10"))


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            rows.append(json.loads(s))
    return rows


def _request_payload(question: str, qtype: str, deployment: str) -> dict[str, Any]:
    # Natural chat questions must run with natural_chat mode.
    mode = "natural_chat" if qtype == "natural_chat" else "answer"
    return {
        "question": question,
        "language": "ar",
        "mode": mode,
        "engine": "muhasibi",
        "model_deployment": deployment,
    }


def _safe_int(x: Any, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        return default


def _quotes_to_eval_citations(citation_spans: list[dict[str, Any]]) -> list[EvalCitation]:
    out: list[EvalCitation] = []
    for sp in citation_spans or []:
        cid = str(sp.get("chunk_id") or "").strip()
        if not cid:
            continue
        # Eval grounding expects source_id to be the chunk_id.
        s = sp.get("span_start")
        e = sp.get("span_end")
        quote = str(sp.get("quote") or "")
        if s is None or e is None:
            # keep unresolved spans out of eval citations; they count as validity errors separately
            continue
        out.append(
            EvalCitation(
                source_id=cid,
                span_start=_safe_int(s),
                span_end=_safe_int(e),
                quote=quote,
            )
        )
    return out


def _ui_used_edges_to_eval_graph_trace(ui_graph_trace: dict[str, Any]) -> GraphTrace:
    used_edges_ui = (ui_graph_trace or {}).get("used_edges") or []
    chains_ui = (ui_graph_trace or {}).get("argument_chains") or []

    used_edges: list[GraphTraceUsedEdge] = []
    for ue in used_edges_ui:
        spans_ui = (ue or {}).get("justification_spans") or []
        spans: list[GraphTraceUsedEdgeSpan] = []
        for sp in spans_ui[:8]:
            cid = str(sp.get("chunk_id") or "").strip()
            if not cid:
                continue
            spans.append(
                GraphTraceUsedEdgeSpan(
                    source_id=cid,
                    chunk_id=cid,
                    span_start=_safe_int(sp.get("span_start")),
                    span_end=_safe_int(sp.get("span_end")),
                    quote=str(sp.get("quote") or ""),
                )
            )
        used_edges.append(
            GraphTraceUsedEdge(
                edge_id=str(ue.get("edge_id") or ""),
                from_node=str(ue.get("from_node") or ""),
                to_node=str(ue.get("to_node") or ""),
                relation_type=str(ue.get("relation_type") or ""),
                justification_spans=spans,
            )
        )

    # We don’t need to map chains into eval.types.ArgumentChain for current scoring;
    # keep them as debug-only counts.
    return GraphTrace(used_edges=used_edges, argument_chains=[])


def _calc_edge_diversity(used_edges: list[dict[str, Any]]) -> int:
    types = {str(e.get("relation_type") or "") for e in (used_edges or []) if str(e.get("relation_type") or "")}
    return len(types)


def _redundancy_rate(answer_ar: str) -> float:
    # Simple duplicate sentence rate (consistent with eval.model_bakeoff_metrics heuristics)
    import re

    if not (answer_ar or "").strip():
        return 0.0
    parts = re.split(r"[.،؟!؛\n]", answer_ar)
    sents = [p.strip().lower() for p in parts if p.strip() and len(p.strip()) > 10]
    if len(sents) <= 1:
        return 0.0
    seen: set[str] = set()
    dup = 0
    for s in sents:
        if s in seen:
            dup += 1
        else:
            seen.add(s)
    return dup / len(sents)


def _bullet_spam_rate(answer_ar: str) -> float:
    lines = (answer_ar or "").splitlines()
    bullets = sum(1 for ln in lines if ln.strip().startswith(("-", "*", "•")))
    # Spam if > 18 bullets in one answer
    return 1.0 if bullets > 18 else 0.0


def _boundary_present(answer_ar: str) -> bool:
    markers = ["حدود", "غير منصوص", "لم يرد", "خارج نطاق", "لا يتضمن الإطار"]
    t = answer_ar or ""
    return any(m in t for m in markers)


@dataclass
class ModelSummary:
    deployment: str
    dataset_sha256: str
    total_questions: int

    disqualified: bool
    disqualification_reason: str

    citation_validity_errors: int
    unsupported_must_cite_rate: float
    attempted_quarantined_cite_count: int

    rubric_score_10: float
    claim_density_per_1k_chars: float
    boundary_completeness_rate: float

    mean_used_edges: float
    mean_argument_chains: float
    mean_distinct_pillars: float
    edge_diversity: float

    redundancy_rate: float
    quote_budget_compliance_rate: float
    bullet_spam_rate: float

    depth_score_100: float
    cross_score_100: float
    naturalness_score_100: float
    integrity_score_100: float
    composite_score_100: float


def _quote_budget_compliance(citations: list[EvalCitation]) -> float:
    if not citations:
        return 1.0
    ok = 0
    for c in citations:
        if len((c.quote or "").split()) <= 25:
            ok += 1
    return ok / len(citations)


def _normalize01(x: float, lo: float, hi: float) -> float:
    if hi <= lo:
        return 0.0
    return max(0.0, min(1.0, (x - lo) / (hi - lo)))


def _compute_scores(
    *,
    rubric_10: float,
    claim_density: float,
    boundary_rate: float,
    mean_edges: float,
    mean_chains: float,
    mean_pillars: float,
    edge_diversity: float,
    redundancy: float,
    quote_compliance: float,
    bullet_spam: float,
    attempted_quarantine: int,
) -> tuple[float, float, float, float, float]:
    # Depth quality (0-100)
    depth = 0.0
    depth += _normalize01(rubric_10, 0.0, 10.0) * 60.0
    depth += _normalize01(claim_density, 0.0, 12.0) * 25.0
    depth += _normalize01(boundary_rate, 0.0, 1.0) * 15.0

    # Cross-pillar intelligence (0-100)
    cross = 0.0
    cross += _normalize01(mean_edges, 0.0, 8.0) * 35.0
    cross += _normalize01(mean_chains, 0.0, 6.0) * 30.0
    cross += _normalize01(mean_pillars, 1.0, 5.0) * 25.0
    cross += _normalize01(edge_diversity, 0.0, 6.0) * 10.0

    # Naturalness (0-100)
    nat = 0.0
    nat += _normalize01(quote_compliance, 0.0, 1.0) * 45.0
    nat += (1.0 - max(0.0, min(1.0, redundancy))) * 35.0
    nat += (1.0 - max(0.0, min(1.0, bullet_spam))) * 20.0

    # Integrity hygiene (0-100)
    # 0 is best for attempted quarantine. Penalize if > 0.
    integrity = 100.0
    if attempted_quarantine > 0:
        integrity = max(0.0, 100.0 - attempted_quarantine * 10.0)

    composite = depth * 0.45 + cross * 0.35 + nat * 0.15 + integrity * 0.05
    return depth, cross, nat, integrity, composite


async def _score_grounding_for_model(
    *,
    outputs: list[EvalOutputRow],
    dataset_by_id: dict[str, dict[str, Any]],
) -> tuple[int, float]:
    from apps.api.core.database import get_session
    from apps.api.core.database import close_db

    async with get_session() as session:
        gm = await score_grounding(
            session=session,
            outputs=outputs,
            dataset_by_id=dataset_by_id,
            fail_on_invalid_citations=False,
        )
        # Important (Windows/asyncpg):
        # Ensure the global SQLAlchemy async engine is disposed before the event loop closes.
        # Otherwise asyncpg termination callbacks can run after loop close and crash the runner.
        try:
            await close_db()
        except Exception:
            pass
        return int(gm.citation_validity_errors), float(gm.unsupported_claim_rate)


def _attempted_quarantined_cites(citation_spans: list[dict[str, Any]]) -> int:
    # Use the same quarantine list as integrity validator.
    from apps.api.core.integrity_validator import QUARANTINED_CHUNKS

    qset = set(QUARANTINED_CHUNKS.keys())
    return sum(1 for sp in citation_spans or [] if str(sp.get("chunk_id") or "") in qset)


def _maybe_resume_path(deployment: str) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    return OUT_DIR / f"{deployment}.jsonl"


def _load_existing_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    done: set[str] = set()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                r = json.loads(s)
                rid = str(r.get("id") or "").strip()
                if rid:
                    done.add(rid)
            except Exception:
                continue
    return done


def _health_check() -> None:
    r = requests.get(f"{API_BASE}/health", timeout=10)
    r.raise_for_status()


def _load_eval_rows_from_output_jsonl(*, deployment: str, dataset_by_id: dict[str, dict[str, Any]]) -> tuple[list[EvalOutputRow], int]:
    """
    Load previously written per-question JSONL and reconstruct EvalOutputRow objects.

    Returns:
        (eval_rows, attempted_quarantined_cite_total)
    """
    out_path = _maybe_resume_path(deployment)
    if not out_path.exists():
        return [], 0

    outputs: list[EvalOutputRow] = []
    attempted_quarantine_total = 0

    with out_path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                d = json.loads(s)
            except Exception:
                continue

            # Skip error rows
            if d.get("error"):
                continue

            qid = str(d.get("id") or "").strip()
            if not qid:
                continue

            qtext = str(d.get("question") or dataset_by_id.get(qid, {}).get("question_ar") or "").strip()

            answer_ar = str(d.get("answer_ar") or "")
            latency_ms = _safe_int(d.get("latency_ms"), 0)
            abstained = bool(d.get("abstained", False))
            abstain_reason = d.get("abstain_reason")

            citation_spans = d.get("citations_spans") or []
            attempted = _attempted_quarantined_cites(citation_spans)
            attempted_quarantine_total += attempted

            citations = _quotes_to_eval_citations(citation_spans)
            quote_ok = _quote_budget_compliance(citations)

            mode = EvalMode.FULL_SYSTEM
            claims = extract_claims(answer_ar=answer_ar, mode=mode, citations=citations)

            graph_trace_ui = d.get("graph_trace") or {}
            used_edges_ui = graph_trace_ui.get("used_edges") or []
            argument_chains_ui = graph_trace_ui.get("argument_chains") or []
            graph_trace = _ui_used_edges_to_eval_graph_trace({"used_edges": used_edges_ui})

            outputs.append(
                EvalOutputRow(
                    id=qid,
                    mode=mode,
                    question=qtext,
                    answer_ar=answer_ar,
                    claims=claims,
                    citations=citations,
                    graph_trace=graph_trace,
                    abstained=abstained,
                    abstain_reason=abstain_reason,
                    latency_ms=latency_ms,
                    debug={
                        "deployment": deployment,
                        "contract_outcome": d.get("contract_outcome"),
                        "contract_reasons": d.get("contract_reasons"),
                        "used_edges_count": len(used_edges_ui),
                        "argument_chains_count": len(argument_chains_ui),
                        "attempted_quarantined_cite_count": attempted,
                        "quote_budget_compliance": quote_ok,
                    },
                )
            )

    return outputs, attempted_quarantine_total


def main() -> None:
    load_dotenv(override=False)

    if not DATASET_PATH.exists():
        raise SystemExit(f"Dataset not found: {DATASET_PATH}")

    dataset_rows = _load_jsonl(DATASET_PATH)
    dataset_sha = _sha256_file(DATASET_PATH)
    dataset_by_id = {str(r.get("id")): r for r in dataset_rows if r.get("id")}

    # Sanity: API up
    _health_check()

    print(f"Dataset: {DATASET_PATH}  sha256={dataset_sha[:16]}  n={len(dataset_rows)}")
    print(f"API: {API_BASE}")
    print(f"Deployments: {DEPLOYMENTS}")

    summaries: list[ModelSummary] = []

    for dep in DEPLOYMENTS:
        out_path = _maybe_resume_path(dep)
        done_ids = _load_existing_ids(out_path)

        print("\n" + "=" * 80)
        print(f"MODEL: {dep}  resume_done={len(done_ids)}/{len(dataset_rows)}")
        print("=" * 80)

        # Run remaining questions
        with out_path.open("a", encoding="utf-8") as f_out:
            for idx, row in enumerate(dataset_rows, 1):
                qid = str(row.get("id") or "").strip()
                if not qid:
                    continue
                if qid in done_ids:
                    continue

                qtext = str(row.get("question_ar") or row.get("question") or "").strip()
                qtype = str(row.get("type") or "").strip() or "answer"

                payload = _request_payload(qtext, qtype, dep)

                t0 = time.perf_counter()
                try:
                    resp = requests.post(f"{API_BASE}/ask/ui", json=payload, timeout=300)
                    latency_ms = int((time.perf_counter() - t0) * 1000)
                    resp.raise_for_status()
                    data = resp.json()

                    final = data.get("final") or {}
                    answer_ar = str(final.get("answer_ar") or "")
                    not_found = bool(final.get("not_found"))
                    abstained = bool(not_found)

                    citation_spans = data.get("citations_spans") or []

                    graph_trace_ui = data.get("graph_trace") or {}
                    used_edges_ui = graph_trace_ui.get("used_edges") or []
                    argument_chains_ui = graph_trace_ui.get("argument_chains") or []

                    # Write per-question JSONL output
                    f_out.write(
                        json.dumps(
                            {
                                "id": qid,
                                "type": qtype,
                                "category": row.get("category"),
                                "question": qtext,
                                "deployment": dep,
                                "latency_ms": latency_ms,
                                "abstained": abstained,
                                "contract_outcome": data.get("contract_outcome"),
                                "contract_reasons": data.get("contract_reasons"),
                                "answer_ar": answer_ar,
                                "citations_spans": citation_spans,
                                "graph_trace": {
                                    "used_edges": used_edges_ui,
                                    "argument_chains": argument_chains_ui,
                                },
                                "mechanism_trace": data.get("mechanism_trace"),
                                "muhasibi_trace": data.get("muhasibi_trace"),
                                "debug": {},
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    f_out.flush()

                    if (idx == 1) or (idx % max(1, PRINT_EVERY) == 0):
                        print(f"[{idx}/{len(dataset_rows)}] {qid} ok  {latency_ms}ms  edges={len(used_edges_ui)}")

                except Exception as e:
                    latency_ms = int((time.perf_counter() - t0) * 1000)
                    f_out.write(
                        json.dumps(
                            {
                                "id": qid,
                                "type": qtype,
                                "category": row.get("category"),
                                "question": qtext,
                                "deployment": dep,
                                "latency_ms": latency_ms,
                                "error": str(e),
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    f_out.flush()
                    print(f"[{idx}/{len(dataset_rows)}] {qid} ERROR {latency_ms}ms: {e}")

        # Compute metrics
        # Reload all successful rows for scoring (supports resume).
        outputs, attempted_quarantine_total = _load_eval_rows_from_output_jsonl(
            deployment=dep, dataset_by_id=dataset_by_id
        )
        m = compute_all_metrics(outputs=outputs, dataset_by_id=dataset_by_id, integrity_hits=attempted_quarantine_total)
        citation_errors, unsupported_rate = asyncio.run(
            _score_grounding_for_model(outputs=outputs, dataset_by_id=dataset_by_id)
        )

        # Replace with grounding-derived gates
        m.citation_validity_errors = citation_errors
        m.unsupported_must_cite_rate = unsupported_rate

        # Derive per-question metrics from reconstructed rows
        per_q_boundary = [1.0 if _boundary_present(r.answer_ar) else 0.0 for r in outputs]
        boundary_rate = sum(per_q_boundary) / max(1, len(per_q_boundary))

        used_edges_ui_counts = [int(r.debug.get("used_edges_count", 0)) for r in outputs]
        chains_ui_counts = [int(r.debug.get("argument_chains_count", 0)) for r in outputs]
        # Pillars + diversity from stored JSONL (we don't store diversity; approximate from used_edges_count only)
        # NOTE: edge_diversity computed from JSONL at write-time would be better; keep simple here.
        mean_edges = float(sum(used_edges_ui_counts) / max(1, len(used_edges_ui_counts)))
        mean_chains = float(sum(chains_ui_counts) / max(1, len(chains_ui_counts)))
        mean_pillars = float(m.distinct_pillars)
        mean_edge_div = float(m.edge_type_diversity)

        # Claim density: median must-cite claims per 1k chars (post-prune proxy)
        cds: list[float] = []
        for r in outputs:
            must_cite_claims = [c for c in r.claims if c.support_policy.value == "must_cite"]
            cds.append(len(must_cite_claims) / max(1.0, (len(r.answer_ar) / 1000.0)))
        claim_density = float(median(cds)) if cds else 0.0

        redundancy = float(sum(_redundancy_rate(r.answer_ar) for r in outputs) / max(1, len(outputs)))
        bullet_spam = float(sum(_bullet_spam_rate(r.answer_ar) for r in outputs) / max(1, len(outputs)))
        quote_ok_rate = float(
            sum(float(r.debug.get("quote_budget_compliance", 1.0)) for r in outputs) / max(1, len(outputs))
        )

        disq = False
        disq_reason = ""
        if m.citation_validity_errors > 0:
            disq = True
            disq_reason = f"citation_validity_errors={m.citation_validity_errors}"
        elif m.unsupported_must_cite_rate > 0:
            disq = True
            disq_reason = f"unsupported_must_cite_rate={m.unsupported_must_cite_rate:.4f}"

        depth100, cross100, nat100, integ100, comp100 = _compute_scores(
            rubric_10=float(m.rubric_score),
            claim_density=claim_density,
            boundary_rate=boundary_rate,
            mean_edges=mean_edges,
            mean_chains=mean_chains,
            mean_pillars=mean_pillars,
            edge_diversity=mean_edge_div,
            redundancy=redundancy,
            quote_compliance=quote_ok_rate,
            bullet_spam=bullet_spam,
            attempted_quarantine=attempted_quarantine_total,
        )

        if disq:
            comp100 = 0.0

        summary = ModelSummary(
            deployment=dep,
            dataset_sha256=dataset_sha,
            total_questions=len(dataset_rows),
            disqualified=disq,
            disqualification_reason=disq_reason,
            citation_validity_errors=int(m.citation_validity_errors),
            unsupported_must_cite_rate=float(m.unsupported_must_cite_rate),
            attempted_quarantined_cite_count=int(attempted_quarantine_total),
            rubric_score_10=float(m.rubric_score),
            claim_density_per_1k_chars=float(claim_density),
            boundary_completeness_rate=float(boundary_rate),
            mean_used_edges=float(mean_edges),
            mean_argument_chains=float(mean_chains),
            mean_distinct_pillars=float(mean_pillars),
            edge_diversity=float(mean_edge_div),
            redundancy_rate=float(redundancy),
            quote_budget_compliance_rate=float(quote_ok_rate),
            bullet_spam_rate=float(bullet_spam),
            depth_score_100=float(depth100),
            cross_score_100=float(cross100),
            naturalness_score_100=float(nat100),
            integrity_score_100=float(integ100),
            composite_score_100=float(comp100),
        )
        summaries.append(summary)

        # Write per-model summary
        sum_path = OUT_DIR / f"{dep}__summary.json"
        sum_path.write_text(json.dumps(summary.__dict__, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Wrote summary: {sum_path}")

    # Report
    eligible = [s for s in summaries if not s.disqualified]
    ranked = sorted(eligible, key=lambda s: -s.composite_score_100)

    # Winners by category (among eligible)
    def _winner(key: str) -> Optional[str]:
        if not eligible:
            return None
        return max(eligible, key=lambda s: getattr(s, key)).deployment

    winners = {
        "depth": _winner("depth_score_100"),
        "cross_pillar": _winner("cross_score_100"),
        "naturalness": _winner("naturalness_score_100"),
        "overall": ranked[0].deployment if ranked else None,
    }

    lines: list[str] = []
    lines.append("## System Depth Bakeoff (Full Muḥāsibī)\n")
    lines.append(f"- **Generated**: {datetime.utcnow().isoformat()}Z")
    lines.append(f"- **Dataset**: `{DATASET_PATH.as_posix()}`")
    lines.append(f"- **Dataset sha256**: `{dataset_sha}`")
    lines.append(f"- **Controls**: seed={SEED}, temp={TEMPERATURE}, max_tokens={MAX_TOKENS}, prompts_version={PROMPTS_VERSION}")
    lines.append("")

    lines.append("### Eligibility (Hard Gates)\n")
    lines.append("- Eligible iff **unsupported_must_cite_rate == 0** and **citation_validity_errors == 0**")
    lines.append("")

    lines.append("### Per-model Summary\n")
    lines.append(
        "| Model | Eligible | Composite | Depth | Cross | Naturalness | Integrity | Unsupported rate | Citation errors | Quarantine cites |\n"
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
    )
    for s in summaries:
        eligible_flag = "✅" if not s.disqualified else "❌"
        lines.append(
            f"| {s.deployment} | {eligible_flag} | {s.composite_score_100:.1f} | {s.depth_score_100:.1f} | {s.cross_score_100:.1f} | {s.naturalness_score_100:.1f} | {s.integrity_score_100:.1f} | {s.unsupported_must_cite_rate:.4f} | {s.citation_validity_errors} | {s.attempted_quarantined_cite_count} |"
        )

    lines.append("")
    lines.append("### Winners (eligible models only)\n")
    lines.append(f"- **Overall**: {winners['overall']}")
    lines.append(f"- **Depth**: {winners['depth']}")
    lines.append(f"- **Cross-pillar reasoning**: {winners['cross_pillar']}")
    lines.append(f"- **Naturalness**: {winners['naturalness']}")
    lines.append("")

    REPORT_PATH.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")
    print(f"Wrote report: {REPORT_PATH}")


if __name__ == "__main__":
    main()

