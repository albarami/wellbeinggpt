"""Check edge scorer training data."""
import json

count = 0
pos = 0
neg = 0

with open("data/phase2/edge_scorer_train.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        count += 1
        data = json.loads(line)
        if data.get("label") == 1:
            pos += 1
        else:
            neg += 1
        if count <= 3:
            print(f"Example {count}:")
            print(f"  nodeA: {str(data.get('nodeA_text', ''))[:60]}...")
            print(f"  nodeB: {str(data.get('nodeB_text', ''))[:60]}...")
            print(f"  relation: {data.get('relation_type')}")
            print(f"  justification: {str(data.get('justification_span', ''))[:60]}...")
            print(f"  label: {data.get('label')}")
            print()

print(f"Total edge scorer pairs: {count}")
print(f"Positives: {pos}")
print(f"Negatives: {neg}")
