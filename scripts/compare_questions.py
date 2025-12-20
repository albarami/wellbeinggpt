"""Compare questions."""
import json
from pathlib import Path

# Hardcoded question (what worked earlier)
q1 = "ما المنظور الكلي للإطار؟"

# Load from dataset
REPO = Path(__file__).resolve().parents[1]
DATASET = REPO / "eval/datasets/regression_unexpected_fails.jsonl"
rows = [json.loads(l) for l in DATASET.read_text(encoding="utf-8").splitlines() if l.strip()]
synth = [r for r in rows if r["id"] == "synth-006"][0]
q2 = synth.get("question_ar", "")

print(f"Q1 len: {len(q1)}")
print(f"Q2 len: {len(q2)}")
print(f"Same: {q1 == q2}")
print(f"Q1 bytes: {q1.encode('utf-8')[:50]}")
print(f"Q2 bytes: {q2.encode('utf-8')[:50]}")
