"""Retrieval quality scoring."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from eval.types import EvalOutputRow


@dataclass(frozen=True)
class RetrievalMetrics:
    recall_at_k: float
    mrr: float


def score_retrieval(
    *,
    outputs: list[EvalOutputRow],
    dataset_by_id: dict[str, dict[str, Any]],
    k: int = 10,
) -> RetrievalMetrics:
    recalls: list[float] = []
    rr: list[float] = []

    for r in outputs:
        d = dataset_by_id.get(r.id, {})
        required = [str(x) for x in (d.get("required_evidence_refs") or []) if str(x)]
        if not required:
            continue

        top = [t.chunk_id for t in (r.retrieval_trace.top_k_chunks or [])][:k]
        if not top:
            recalls.append(0.0)
            rr.append(0.0)
            continue

        hit = len(set(required).intersection(set(top)))
        recalls.append(hit / max(len(set(required)), 1))

        # reciprocal rank
        best = 0.0
        for i, cid in enumerate(top, start=1):
            if cid in required:
                best = 1.0 / i
                break
        rr.append(best)

    return RetrievalMetrics(
        recall_at_k=(sum(recalls) / len(recalls)) if recalls else 0.0,
        mrr=(sum(rr) / len(rr)) if rr else 0.0,
    )
