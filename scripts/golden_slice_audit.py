"""Golden-slice spot audit (deterministic).

This script is a quick sanity checker to support the human 20-question audit:
- verifies deep-mode required section headers are present when expected
- verifies answered rows contain citations
- verifies cross-pillar rows contain a path trace (graph_trace)

It does NOT attempt to judge writing quality; it only checks structural gates.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REQUIRED_DEEP_HEADERS = [
    "تعريف المفهوم داخل الإطار",
    "التأصيل والأدلة (مختصر ومركز)",
    "الربط بين الركائز (مع سبب الربط)",
    "تطبيق عملي على الحالة/السؤال",
    "تنبيهات وأخطاء شائعة",
    "خلاصة تنفيذية (3 نقاط)",
]


@dataclass(frozen=True)
class RowAudit:
    id: str
    ok: bool
    problems: list[str]


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = (line or "").strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _looks_like_deep_required(question: str, answer: str) -> bool:
    q = (question or "")
    a = (answer or "")
    if any(h in a for h in REQUIRED_DEEP_HEADERS):
        return True
    if "وبالترتيب التالي" in q and "تعريف المفهوم" in q:
        return True
    return False


def audit_rows(*, dataset_type: str, outputs: list[dict[str, Any]], limit: int) -> list[RowAudit]:
    out: list[RowAudit] = []
    for r in outputs[: max(0, int(limit))]:
        rid = str(r.get("id") or "")
        answer = str(r.get("answer_ar") or "")
        question = str(r.get("question") or "")
        abstained = bool(r.get("abstained"))
        citations = list(r.get("citations") or [])
        graph_trace = dict(r.get("graph_trace") or {})

        problems: list[str] = []

        if not abstained:
            if len(citations) <= 0:
                problems.append("NO_CITATIONS")

        if dataset_type == "cross":
            edges = list(graph_trace.get("edges") or [])
            nodes = list(graph_trace.get("nodes") or [])
            paths = list(graph_trace.get("paths") or [])
            if not (edges or paths):
                problems.append("MISSING_GRAPH_TRACE")
            if not nodes:
                problems.append("MISSING_GRAPH_NODES")

        if dataset_type == "deep":
            if _looks_like_deep_required(question, answer):
                missing = [h for h in REQUIRED_DEEP_HEADERS if h not in answer]
                if missing:
                    problems.append("MISSING_HEADERS:" + ",".join(missing))

        out.append(RowAudit(id=rid, ok=(len(problems) == 0), problems=problems))

    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-type", required=True, choices=["gold", "cross", "negative", "injection", "deep"])
    p.add_argument("--outputs", required=True, help="Path to FULL_SYSTEM output jsonl")
    p.add_argument("--limit", type=int, default=20)
    args = p.parse_args()

    out_path = Path(args.outputs)
    rows = _read_jsonl(out_path)
    audits = audit_rows(dataset_type=str(args.dataset_type), outputs=rows, limit=int(args.limit))

    bad = [a for a in audits if not a.ok]
    print(
        json.dumps(
            {
                "file": str(out_path.as_posix()),
                "dataset_type": args.dataset_type,
                "checked": len(audits),
                "failed": len(bad),
                "failures": [{"id": b.id, "problems": b.problems} for b in bad[:20]],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
