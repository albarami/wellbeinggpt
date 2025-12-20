"""Test the trained reranker model."""
import os
import json

os.environ["RERANKER_ENABLED"] = "true"
os.environ["RERANKER_MODEL_PATH"] = "checkpoints/reranker/final"

from apps.api.retrieve.reranker import create_reranker_from_env

rr = create_reranker_from_env()
print(f"Reranker enabled: {rr.is_enabled()}")
print(f"Type: {type(rr).__name__}")

# Test scoring with Arabic text
query = "ما هو التوازن في الإسلام؟"
doc_positive = "التوازن يعني الاعتدال في كل شيء من العبادات والمعاملات"
doc_negative = "الطقس اليوم مشمس وجميل"

score_pos = rr.score(query, doc_positive)
score_neg = rr.score(query, doc_negative)

result = {
    "reranker_enabled": rr.is_enabled(),
    "reranker_type": type(rr).__name__,
    "positive_doc_score": score_pos,
    "negative_doc_score": score_neg,
    "difference": score_pos - score_neg,
    "correct_ranking": score_pos > score_neg,
}

# Write to file (avoid console encoding issues)
with open("reranker_test_result.json", "w", encoding="utf-8") as f:
    json.dump(result, f, indent=2, ensure_ascii=False)

print(f"Positive doc score: {score_pos:.4f}")
print(f"Negative doc score: {score_neg:.4f}")
print(f"Difference: {score_pos - score_neg:.4f}")

if score_pos > score_neg:
    print("[OK] Reranker correctly ranks relevant doc higher!")
else:
    print("[WARN] Reranker may need more training")

print("Results saved to reranker_test_result.json")
