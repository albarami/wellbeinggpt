"""Run regression tests on the unexpected failure questions."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import json
import requests
from datetime import datetime

REPO = Path(__file__).resolve().parents[1]
DATASET = REPO / "eval/datasets/regression_unexpected_fails.jsonl"
OUT_DIR = REPO / "eval/output/regression"
BASE_URL = "http://127.0.0.1:8000"
DEPLOYMENT = "gpt-5.1"

OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_dataset():
    rows = []
    with DATASET.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                rows.append(json.loads(s))
    return rows


def run_question(row: dict, max_retries: int = 2) -> dict:
    """Run a single question through the API with retry on failure."""
    import time
    
    qid = row.get("id", "unknown")
    question = row.get("question_ar") or row.get("question", "")
    qtype = row.get("type", "")
    
    # Natural chat questions must run with natural_chat mode
    mode = "natural_chat" if qtype == "natural_chat" else "answer"
    
    for attempt in range(max_retries + 1):
        try:
            response = requests.post(
                f"{BASE_URL}/ask/ui",
                json={
                    "question": question,
                    "model_deployment": DEPLOYMENT,
                    "mode": mode,
                },
                timeout=180,
            )
            
            if response.status_code == 200:
                data = response.json()
                result = {
                    "id": qid,
                    "type": qtype,
                    "mode_used": data.get("mode_used"),
                    "contract_outcome": data.get("contract_outcome"),
                    "citations_count": len(data.get("citations_spans") or []),
                    "edges_count": len((data.get("graph_trace") or {}).get("used_edges") or []),
                    "abstained": data.get("abstain_reason") is not None,
                    "success": data.get("contract_outcome") in ("PASS_FULL", "PASS_PARTIAL"),
                }
                
                # Retry once if failed with 0 citations (possible cold-start issue)
                if not result["success"] and result["citations_count"] == 0 and attempt < max_retries:
                    time.sleep(2)  # Brief pause before retry
                    continue
                    
                return result
            else:
                if attempt < max_retries:
                    time.sleep(2)
                    continue
                return {
                    "id": qid,
                    "type": qtype,
                    "error": f"HTTP {response.status_code}",
                    "success": False,
                }
        except Exception as e:
            if attempt < max_retries:
                time.sleep(2)
                continue
            return {
                "id": qid,
                "type": qtype,
                "error": str(e),
                "success": False,
            }
    
    # Should not reach here
    return {"id": qid, "type": qtype, "error": "max_retries_exceeded", "success": False}


def main():
    print(f"Regression Test - {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"Dataset: {DATASET}")
    print(f"Deployment: {DEPLOYMENT}")
    print("=" * 60)
    
    rows = load_dataset()
    print(f"Loaded {len(rows)} questions")
    print()
    
    results = []
    passed = 0
    failed = 0
    
    for i, row in enumerate(rows):
        qid = row.get("id", f"q{i}")
        print(f"[{i+1}/{len(rows)}] {qid}...", end=" ", flush=True)
        
        result = run_question(row)
        results.append(result)
        
        if result.get("success"):
            passed += 1
            status = "PASS"
        else:
            failed += 1
            status = "FAIL"
        
        outcome = result.get("contract_outcome", result.get("error", "?"))
        cites = result.get("citations_count", 0)
        print(f"{status} ({outcome}, cites={cites})")
    
    print()
    print("=" * 60)
    print(f"RESULTS: {passed}/{len(rows)} passed, {failed} failed")
    print(f"Pass rate: {100*passed/len(rows):.1f}%")
    
    # Write results
    out_file = OUT_DIR / f"{DEPLOYMENT}_regression.json"
    out_file.write_text(json.dumps({
        "timestamp": datetime.now().isoformat(),
        "dataset": str(DATASET),
        "deployment": DEPLOYMENT,
        "total": len(rows),
        "passed": passed,
        "failed": failed,
        "pass_rate": passed / len(rows) if rows else 0,
        "results": results,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nWrote: {out_file}")
    
    # Exit with error if any failed
    if failed > 0:
        print(f"\nWARNING: {failed} questions failed regression")
        sys.exit(1)
    else:
        print("\nAll regression tests passed!")
        sys.exit(0)


if __name__ == "__main__":
    main()

