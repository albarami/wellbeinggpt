"""Export reranker ON failures for analysis."""
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
ON_FILE = REPO / "eval/output/bakeoff_ab/reranker_on.jsonl"
OFF_FILE = REPO / "eval/output/bakeoff_ab/reranker_off.jsonl"
DATASET = REPO / "eval/datasets/bakeoff_depth_v1.jsonl"

# Load results
on_results = [json.loads(l) for l in ON_FILE.read_text(encoding="utf-8").splitlines() if l.strip()]
off_results = [json.loads(l) for l in OFF_FILE.read_text(encoding="utf-8").splitlines() if l.strip()]
dataset = [json.loads(l) for l in DATASET.read_text(encoding="utf-8").splitlines() if l.strip()]

# Create lookup for question info
dataset_by_id = {d.get("id"): d for d in dataset}

# Find failures
on_failures = [r for r in on_results if not r.get("success")]
off_failures = [r for r in off_results if not r.get("success")]

print(f"Reranker ON failures: {len(on_failures)}")
print(f"Reranker OFF failures: {len(off_failures)}")

# Check which failures are unique to ON (not in OFF)
on_failure_ids = {r.get("id") for r in on_failures}
off_failure_ids = {r.get("id") for r in off_failures}

only_on_fails = on_failure_ids - off_failure_ids
only_off_fails = off_failure_ids - on_failure_ids
both_fail = on_failure_ids & off_failure_ids

print(f"Fail ONLY with reranker ON: {len(only_on_fails)} - {list(only_on_fails)}")
print(f"Fail ONLY with reranker OFF: {len(only_off_fails)} - {list(only_off_fails)}")
print(f"Fail in BOTH conditions: {len(both_fail)} - {list(both_fail)}")

# Build detailed failure info
on_failure_details = []
for r in on_failures:
    qid = r.get("id")
    q_info = dataset_by_id.get(qid, {})
    on_failure_details.append({
        "id": qid,
        "type": r.get("type"),
        "outcome": r.get("contract_outcome"),
        "citations": r.get("citations_count", 0),
        "question_ar": q_info.get("question_ar", q_info.get("question", "")),
    })

off_failure_details = []
for r in off_failures:
    qid = r.get("id")
    q_info = dataset_by_id.get(qid, {})
    off_failure_details.append({
        "id": qid,
        "type": r.get("type"),
        "outcome": r.get("contract_outcome"),
        "citations": r.get("citations_count", 0),
        "question_ar": q_info.get("question_ar", q_info.get("question", "")),
    })

# Export detailed failure report
report = {
    "summary": {
        "on_failures_count": len(on_failures),
        "off_failures_count": len(off_failures),
        "only_on_fails": list(only_on_fails),
        "only_off_fails": list(only_off_fails),
        "both_fail": list(both_fail),
    },
    "on_failure_details": on_failure_details,
    "off_failure_details": off_failure_details,
}

(REPO / "eval/output/bakeoff_ab/failure_analysis.json").write_text(
    json.dumps(report, ensure_ascii=False, indent=2),
    encoding="utf-8"
)
print(f"Full report: eval/output/bakeoff_ab/failure_analysis.json")
