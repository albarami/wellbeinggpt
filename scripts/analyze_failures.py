"""Analyze which questions are failing contract across models."""

import json
from pathlib import Path
from collections import defaultdict

OUT_DIR = Path("d:/wellbeingqa/eval/output/bakeoff_depth_v1_system")
DEPLOYMENTS = ["gpt-5-chat", "gpt-5.1", "gpt-5.2"]


def load_jsonl(path: Path) -> list[dict]:
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


def main():
    failures: dict[str, dict[str, str]] = defaultdict(dict)  # qid -> {dep -> outcome}
    
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
            outcome = str(r.get("contract_outcome") or "").upper()
            if outcome not in {"PASS_FULL", "PASS_PARTIAL"}:
                failures[qid][dep] = outcome or "UNKNOWN"
    
    print(f"Questions with FAIL outcomes:\n")
    print(f"{'QID':<20} {'Type':<15} {'gpt-5-chat':<12} {'gpt-5.1':<12} {'gpt-5.2':<12}")
    print("-" * 70)
    
    # Get question types
    for qid in sorted(failures.keys()):
        # Find type from any model's output
        qtype = "?"
        for dep in DEPLOYMENTS:
            path = OUT_DIR / f"{dep}.jsonl"
            for r in load_jsonl(path):
                if r.get("id") == qid:
                    qtype = str(r.get("type") or r.get("category") or "?")
                    break
            if qtype != "?":
                break
        
        outcomes = [failures[qid].get(dep, "-") for dep in DEPLOYMENTS]
        print(f"{qid:<20} {qtype:<15} {outcomes[0]:<12} {outcomes[1]:<12} {outcomes[2]:<12}")
    
    print()
    print(f"Total failing questions: {len(failures)}")
    
    # Count by type
    type_counts: dict[str, int] = defaultdict(int)
    for qid in failures:
        for dep in DEPLOYMENTS:
            path = OUT_DIR / f"{dep}.jsonl"
            for r in load_jsonl(path):
                if r.get("id") == qid:
                    qtype = str(r.get("type") or r.get("category") or "unknown")
                    type_counts[qtype] += 1
                    break
            break
    
    print("\nBy question type:")
    for t, c in sorted(type_counts.items(), key=lambda x: -x[1]):
        print(f"  {t}: {c}")


if __name__ == "__main__":
    main()
