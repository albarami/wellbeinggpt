"""Export unexpected failures from bakeoff and create regression dataset.

This script:
1. Identifies all unexpected contract failures across models
2. Exports detailed diagnostics for each
3. Creates eval/datasets/regression_unexpected_fails.jsonl for CI gate
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

OUT_DIR = REPO / "eval/output/bakeoff_depth_v1_system"
DATASET_PATH = REPO / "eval/datasets/bakeoff_depth_v1.jsonl"
REGRESSION_PATH = REPO / "eval/datasets/regression_unexpected_fails.jsonl"

DEPLOYMENTS = ["gpt-5-chat", "gpt-5.1", "gpt-5.2"]


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                try:
                    rows.append(json.loads(s))
                except Exception:
                    pass
    return rows


def is_expected_fail(qtype: str) -> bool:
    return qtype in {"injection", "out_of_scope", "oos"}


def main():
    dataset_rows = load_jsonl(DATASET_PATH)
    dataset_by_id = {str(r.get("id")): r for r in dataset_rows}
    
    # Collect all unexpected failures
    unexpected: dict[str, dict[str, Any]] = {}  # qid -> diagnostics
    
    for dep in DEPLOYMENTS:
        path = OUT_DIR / f"{dep}.jsonl"
        rows = load_jsonl(path)
        
        # Deduplicate
        by_id = {}
        for r in rows:
            qid = str(r.get("id") or "")
            if qid:
                by_id[qid] = r
        
        for qid, r in by_id.items():
            qtype = str(r.get("type") or dataset_by_id.get(qid, {}).get("type") or "").lower()
            outcome = str(r.get("contract_outcome") or "").upper()
            
            # Unexpected = not PASS_FULL/PARTIAL and not expected fail type
            if outcome not in {"PASS_FULL", "PASS_PARTIAL"} and not is_expected_fail(qtype):
                if qid not in unexpected:
                    ds_row = dataset_by_id.get(qid, {})
                    # question_ar might be stored as 'question_ar' or 'question'
                    q_text = ds_row.get("question_ar") or ds_row.get("question") or ""
                    unexpected[qid] = {
                        "id": qid,
                        "question_ar": q_text,
                        "type": qtype,
                        "category": ds_row.get("category", ""),
                        "models_failing": [],
                        "diagnostics": {},
                    }
                
                # Add model-specific diagnostics
                unexpected[qid]["models_failing"].append(dep)
                unexpected[qid]["diagnostics"][dep] = {
                    "contract_outcome": outcome,
                    "contract_reasons": r.get("contract_reasons") or [],
                    "mode": r.get("mode") or r.get("type"),
                    "abstained": r.get("abstained", False),
                    "citations_count": len(r.get("citations_spans") or []),
                    "edges_count": len((r.get("graph_trace") or {}).get("used_edges") or []),
                    "answer_snippet": (r.get("answer_ar") or "")[:200],
                }
    
    # Print diagnostics (ASCII only to avoid Windows console encoding issues)
    print(f"Unexpected Failures: {len(unexpected)} questions\n")
    print("=" * 80)
    
    for qid, info in sorted(unexpected.items()):
        print(f"\n{qid} ({info['type']})")
        print(f"  Failing on: {', '.join(info['models_failing'])}")
        for dep, diag in info["diagnostics"].items():
            print(f"  [{dep}]")
            print(f"    outcome={diag['contract_outcome']}")
            print(f"    citations={diag['citations_count']}, edges={diag['edges_count']}, abstained={diag['abstained']}")
    
    # Create regression dataset
    regression_rows = []
    for qid, info in unexpected.items():
        ds_row = dataset_by_id.get(qid, {})
        q_text = info["question_ar"] or ds_row.get("question_ar") or ds_row.get("question") or ""
        regression_rows.append({
            "id": qid,
            "question_ar": q_text,
            "type": info["type"],
            "category": info["category"],
            "answer_requirements": ds_row.get("answer_requirements", {}),
            "expect_abstain": ds_row.get("expect_abstain", False),
            "regression_reason": f"Unexpected fail on: {', '.join(info['models_failing'])}",
        })
    
    # Write regression dataset
    with REGRESSION_PATH.open("w", encoding="utf-8") as f:
        for row in regression_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    
    print(f"\n{'=' * 80}")
    print(f"Wrote {len(regression_rows)} questions to: {REGRESSION_PATH}")
    print("\nThese questions should be investigated and fixed, then added to CI gate.")


if __name__ == "__main__":
    main()

