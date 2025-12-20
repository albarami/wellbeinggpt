"""Test synth-006 question directly."""
import json
import requests

API_URL = "http://127.0.0.1:8000/ask/ui"
QUESTION = "ما المنظور الكلي للإطار؟"

resp = requests.post(
    API_URL,
    json={
        "question": QUESTION,
        "mode": "answer",
        "model_deployment": "gpt-5.1",
    },
    timeout=120,
)

data = resp.json()

result = {
    "status_code": resp.status_code,
    "contract_outcome": data.get("contract_outcome"),
    "abstained": data.get("final", {}).get("not_found"),
    "citations_count": len(data.get("citations_spans") or []),
    "used_edges_count": len(data.get("graph_trace", {}).get("used_edges") or []),
    "answer_length": len(data.get("final", {}).get("answer_ar") or ""),
    "abstain_reason": data.get("abstain_reason"),
}

with open("synth006_result.json", "w", encoding="utf-8") as f:
    json.dump(result, f, ensure_ascii=False, indent=2)

print(json.dumps(result, ensure_ascii=False, indent=2))
