"""Debug training data preparation."""
import json
from pathlib import Path

input_file = Path("eval/output/wellbeing__v1__464e083154d8__seed1337__pv1__FULL_SYSTEM.jsonl")
rows = [json.loads(l) for l in input_file.read_text(encoding="utf-8").splitlines() if l.strip()]

print(f"Total rows: {len(rows)}")
abstained_count = sum(1 for r in rows if r.get("abstained"))
print(f"Abstained: {abstained_count}")
has_cits = sum(1 for r in rows if r.get("citations") and len(r.get("citations", [])) > 0)
print(f"Has citations: {has_cits}")

# Check first non-abstained row with citations
for r in rows:
    if not r.get("abstained") and r.get("citations"):
        cits = r.get("citations", [])
        print(f"Sample citation keys: {list(cits[0].keys()) if cits else []}")
        rt = r.get("retrieval_trace", {})
        print(f"Sample retrieval_trace keys: {list(rt.keys())}")
        print(f"Sample top_k_chunks count: {len(rt.get('top_k_chunks', []))}")
        
        # Check source_id in citations
        for c in cits[:2]:
            print(f"  Citation source_id: {c.get('source_id')}")
        break
