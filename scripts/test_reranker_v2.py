"""Test the trained reranker model with proper classifier handling."""
import os
import json
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_PATH = "checkpoints/reranker/final"

print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
model.eval()

# Move to GPU if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)
print(f"Model loaded on {device}")

def score_pair(query: str, doc: str) -> float:
    """Score a query-document pair using the trained classifier."""
    inputs = tokenizer(
        query,
        doc,
        truncation=True,
        padding=True,
        max_length=256,
        return_tensors="pt",
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        # Use softmax to get probability of class 1 (relevant)
        probs = torch.softmax(logits, dim=-1)
        return probs[0, 1].item()  # Probability of being relevant

# Test cases
test_cases = [
    {
        "query": "ما هو التوازن في الإسلام؟",
        "doc_positive": "التوازن يعني الاعتدال في كل شيء من العبادات والمعاملات",
        "doc_negative": "الطقس اليوم مشمس وجميل"
    },
    {
        "query": "ما هي ركائز الإطار؟",
        "doc_positive": "يتكون الإطار من خمس ركائز أساسية تشمل البعد الإيماني والعبادي",
        "doc_negative": "السيارة تسير بسرعة على الطريق"
    },
    {
        "query": "كيف يتحقق الازدهار؟",
        "doc_positive": "الازدهار يتحقق بالتوازن بين الركائز الخمس والعمل على تحقيق القيم",
        "doc_negative": "الماء ضروري للحياة"
    }
]

results = []
correct_count = 0

print("\nTesting reranker on query-document pairs:\n")

for i, tc in enumerate(test_cases):
    score_pos = score_pair(tc["query"], tc["doc_positive"])
    score_neg = score_pair(tc["query"], tc["doc_negative"])
    is_correct = score_pos > score_neg
    
    if is_correct:
        correct_count += 1
    
    results.append({
        "test": i + 1,
        "positive_score": score_pos,
        "negative_score": score_neg,
        "difference": score_pos - score_neg,
        "correct": is_correct,
    })
    
    print(f"Test {i+1}: pos={score_pos:.4f}, neg={score_neg:.4f}, diff={score_pos-score_neg:+.4f} {'[OK]' if is_correct else '[FAIL]'}")

accuracy = correct_count / len(test_cases)
print(f"\nAccuracy: {correct_count}/{len(test_cases)} = {accuracy*100:.1f}%")

summary = {
    "tests": results,
    "accuracy": accuracy,
    "model_path": MODEL_PATH,
    "device": str(device),
}

with open("reranker_test_result.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)

print("\nResults saved to reranker_test_result.json")

if accuracy >= 0.67:
    print("[OK] Reranker is working - ready for production!")
else:
    print("[WARN] Reranker accuracy low - may need more training data")
