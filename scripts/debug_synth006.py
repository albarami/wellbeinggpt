"""Debug synth-006 issue."""
import json
import requests
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DATASET = REPO / "eval/datasets/regression_unexpected_fails.jsonl"

# Load from dataset
rows = [json.loads(l) for l in DATASET.read_text(encoding="utf-8").splitlines() if l.strip()]
synth = [r for r in rows if r["id"] == "synth-006"][0]

question = synth.get("question_ar") or synth.get("question", "")
qtype = synth.get("type", "")

print(f"Question ID: synth-006")
print(f"Question Type: {qtype}")
print(f"Question length: {len(question)}")

# Call API exactly like regression script
response = requests.post(
    "http://127.0.0.1:8000/ask/ui",
    json={
        "question": question,
        "model_deployment": "gpt-5.1",
        "mode": "natural_chat" if qtype == "natural_chat" else "answer",
    },
    timeout=180,
)

print(f"\nResponse status: {response.status_code}")

if response.status_code == 200:
    data = response.json()
    result = {
        "contract_outcome": data.get('contract_outcome'),
        "citations_count": len(data.get('citations_spans') or []),
        "abstain_reason": data.get('abstain_reason'),
        "not_found": data.get('final', {}).get('not_found'),
        "answer_length": len(data.get('final', {}).get('answer_ar') or ''),
    }
    print(f"Contract: {result['contract_outcome']}, Cites: {result['citations_count']}, Abstained: {result['not_found']}")
    with open("debug_synth006_result.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print("Result saved to debug_synth006_result.json")
else:
    print(f"Error: HTTP {response.status_code}")
