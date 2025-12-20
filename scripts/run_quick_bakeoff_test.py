"""Quick bakeoff test - first 20 questions with gpt-5.1 only."""

import json
import sys
import io
import time
import requests
from pathlib import Path

# Fix Windows console encoding
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

API_BASE = "http://127.0.0.1:8000"
DATASET_PATH = Path("eval/datasets/bakeoff_depth_v1.jsonl")
DEPLOYMENT = "gpt-5.1"
LIMIT = 20

def main():
    # Load dataset
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        questions = [json.loads(line) for line in f][:LIMIT]
    
    print(f"Quick Bakeoff Test: {len(questions)} questions with {DEPLOYMENT}")
    print("=" * 60)
    
    pass_count = 0
    fail_count = 0
    
    for i, q in enumerate(questions):
        qid = q.get("id", f"q{i}")
        qtext = q.get("question_ar", "")[:50]
        mode = q.get("mode", "answer")
        
        # Use "question" not "question_ar" based on dataset format
        payload = {
            "question": q.get("question"),
            "mode": mode,
            "lang": "ar",
            "engine": "muhasibi",
        }
        
        t0 = time.perf_counter()
        try:
            response = requests.post(f"{API_BASE}/ask/ui", json=payload, timeout=120)
            latency_ms = int((time.perf_counter() - t0) * 1000)
            
            if response.status_code != 200:
                print(f"[{i+1}/{len(questions)}] {qid}: ERROR {response.status_code} - {response.text[:200]}")
                fail_count += 1
                continue
            
            data = response.json()
            contract = data.get("contract_outcome", "N/A")
            cites = len(data.get("citations_spans", []))
            abstained = data.get("abstained") or data.get("abstain_reason")
            
            if contract == "PASS_FULL" or (contract == "PASS_PARTIAL" and cites > 0):
                status = "PASS"
                pass_count += 1
            elif abstained:
                status = "ABSTAIN"
                fail_count += 1
            else:
                status = "FAIL"
                fail_count += 1
            
            print(f"[{i+1}/{len(questions)}] {qid}: {status} ({contract}, cites={cites}, {latency_ms}ms)")
            
        except Exception as e:
            print(f"[{i+1}/{len(questions)}] {qid}: ERROR {str(e)[:50]}")
            fail_count += 1
    
    print("=" * 60)
    print(f"Results: {pass_count}/{len(questions)} passed ({100*pass_count/len(questions):.1f}%)")
    print(f"Failures: {fail_count}")

if __name__ == "__main__":
    main()
