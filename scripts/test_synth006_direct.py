"""Test synth-006 directly through the API."""

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
print(f"Question (first 100 chars): {synth006.get('question_ar', '')[:100]}")

# Send request to API
payload = {
    "question": synth006.get("question_ar"),
    "mode": synth006.get("mode", "answer"),
    "lang": "ar",
    "debug": True,  # Enable debug to see reasoning
}

print("\nSending request to API...")
response = requests.post("http://localhost:8000/ask/ui", json=payload, timeout=120)

if response.status_code != 200:
    print(f"ERROR: Status {response.status_code}")
    print(response.text[:500])
    exit(1)

data = response.json()

print(f"\nResponse received:")
print(f"  Contract outcome: {data.get('contract', {}).get('outcome', 'N/A')}")
print(f"  Mode used: {data.get('mode_used', 'N/A')}")
print(f"  Abstained: {data.get('abstained', 'N/A')}")

citations = data.get("citations", [])
print(f"  Citations count: {len(citations)}")

graph_trace = data.get("graph_trace", {})
used_edges = graph_trace.get("used_edges", [])
print(f"  Used edges: {len(used_edges)}")

# Check for reasoning block
reasoning = data.get("reasoning_block", "")
if reasoning:
    print(f"\n  Reasoning (first 200 chars): {reasoning[:200]}")

# Check intent
intent = data.get("intent", {})
print(f"\n  Intent type: {intent.get('intent_type', 'N/A')}")

# Check account issues
account_issues = data.get("account_issues", [])
if account_issues:
    print(f"  Account issues: {account_issues}")

# Check answer length
answer = data.get("answer_ar", "")
print(f"\n  Answer length: {len(answer)} chars")
print(f"  Answer preview: {answer[:200] if answer else 'EMPTY'}")

print("\n--- Done ---")
