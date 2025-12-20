"""Test synth-006 and dump full response."""

import json
import requests
import sys
import io

# Fix Windows console encoding
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# Load the question
dataset_path = "eval/datasets/regression_unexpected_fails.jsonl"
with open(dataset_path, "r", encoding="utf-8") as f:
    questions = [json.loads(line) for line in f]

synth006 = None
for q in questions:
    if q.get("id") == "synth-006":
        synth006 = q
        break

if not synth006:
    print("synth-006 not found")
    exit(1)

print(f"Question ID: {synth006.get('id')}")
print(f"Type: {synth006.get('type')}")
print(f"Question: {synth006.get('question_ar', '')}")

# Send request to API
payload = {
    "question": synth006.get("question_ar"),
    "mode": synth006.get("mode", "answer"),
    "lang": "ar",
    "debug": True,
}

print("\nSending request to API...")
response = requests.post("http://localhost:8000/ask/ui", json=payload, timeout=120)

print(f"\nStatus: {response.status_code}")

# Dump full response to file
with open("eval/output/synth006_response.json", "w", encoding="utf-8") as f:
    json.dump(response.json(), f, ensure_ascii=False, indent=2)

print("Full response saved to: eval/output/synth006_response.json")

# Print key fields
data = response.json()
print(f"\nKey fields:")
print(f"  answer_ar length: {len(data.get('answer_ar', ''))}")
print(f"  mode_used: {data.get('mode_used')}")
print(f"  contract: {data.get('contract')}")
print(f"  citations count: {len(data.get('citations', []))}")
print(f"  not_found: {data.get('not_found')}")
print(f"  abstained: {data.get('abstained')}")
print(f"  intent: {data.get('intent')}")

# Check for errors or warnings
if 'error' in data:
    print(f"\nERROR: {data['error']}")
if 'warnings' in data:
    print(f"\nWARNINGS: {data['warnings']}")
if 'account_issues' in data:
    print(f"\nACCOUNT ISSUES: {data['account_issues']}")

print("\n--- Done ---")
