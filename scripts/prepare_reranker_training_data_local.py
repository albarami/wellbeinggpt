"""Prepare reranker training pairs from bakeoff outputs (no DB required).

This script extracts (query, doc_text) pairs with labels from bakeoff JSONL outputs.
"""

from __future__ import annotations

import json
from pathlib import Path
import random

REPO = Path(__file__).resolve().parents[1]
OUTPUT_DIR = REPO / "eval/output/bakeoff_depth_v1_system"
OUT_PATH = REPO / "data/reranker/train.jsonl"


def extract_training_pairs():
    """Extract training pairs from bakeoff outputs."""
    pairs = []
    
    # Use gpt-5.1 output (best model)
    input_file = OUTPUT_DIR / "gpt-5.1.jsonl"
    if not input_file.exists():
        print(f"ERROR: {input_file} not found")
        return []
    
    rows = [json.loads(l) for l in input_file.read_text(encoding="utf-8").splitlines() if l.strip()]
    print(f"Loaded {len(rows)} rows from {input_file.name}")
    
    for r in rows:
        # Skip abstained
        if r.get("abstained"):
            continue
        
        # Get question
        question = (r.get("question") or "").strip()
        if not question:
            continue
        
        # Get cited chunks (positives)
        citations = r.get("citations_spans") or []
        pos_chunks = []
        for c in citations:
            chunk_id = c.get("chunk_id") or ""
            quote = (c.get("quote") or "").strip()
            if chunk_id and quote and len(quote) > 20:
                pos_chunks.append({
                    "chunk_id": chunk_id,
                    "text": quote,
                })
        
        if not pos_chunks:
            continue
        
        # Get muhasibi trace for retrieved evidence
        trace = r.get("muhasibi_trace") or []
        neg_chunks = []
        
        for state in trace:
            if state.get("state") == "RETRIEVE":
                # The evidence packets are logged but not directly accessible
                # We'll generate synthetic negatives from other rows
                pass
        
        # Create positive pairs
        for pc in pos_chunks:
            pairs.append({
                "query": question,
                "chunk_id": pc["chunk_id"],
                "text_ar": pc["text"],
                "label": 1,
            })
    
    print(f"Generated {len(pairs)} positive pairs")
    
    # Generate synthetic negatives by pairing questions with citations from other rows
    random.seed(1337)
    all_chunks = [(p["chunk_id"], p["text_ar"]) for p in pairs]
    negatives = []
    
    for p in pairs:
        q = p["query"]
        # Find chunks from different questions
        other_chunks = [(cid, txt) for cid, txt in all_chunks if cid != p["chunk_id"]]
        if other_chunks:
            neg_samples = random.sample(other_chunks, min(3, len(other_chunks)))
            for cid, txt in neg_samples:
                negatives.append({
                    "query": q,
                    "chunk_id": cid,
                    "text_ar": txt,
                    "label": 0,
                })
    
    pairs.extend(negatives)
    print(f"Generated {len(negatives)} negative pairs")
    print(f"Total training pairs: {len(pairs)}")
    
    return pairs


def main():
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    pairs = extract_training_pairs()
    
    if not pairs:
        print("ERROR: No training pairs generated")
        return
    
    # Shuffle
    random.seed(1337)
    random.shuffle(pairs)
    
    # Write
    with OUT_PATH.open("w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    
    print(f"\nWrote {len(pairs)} pairs -> {OUT_PATH}")


if __name__ == "__main__":
    main()
