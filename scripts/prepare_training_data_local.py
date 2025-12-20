"""Prepare reranker training pairs from eval outputs (no DB required).

This script extracts (query, passage) pairs with labels from JSONL outputs:
- Positives: Claim texts with supporting evidence spans (label=1)
- Negatives: Random other claims from different questions (label=0)

Usage:
  python scripts/prepare_training_data_local.py \
    --inputs eval/output/*.jsonl \
    --out data/reranker/train.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def extract_training_pairs(input_paths: list[Path], out_path: Path, neg_per_pos: int = 2) -> int:
    """Extract training pairs from eval output JSONL files."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Collect all positive examples
    positives: list[dict] = []
    all_texts: list[str] = []  # For generating negatives
    
    for path in input_paths:
        if not path.exists():
            continue
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                
                if r.get("abstained"):
                    continue
                
                question = str(r.get("question") or "").strip()
                if not question:
                    continue
                
                # Extract from citations (direct quotes)
                for cit in (r.get("citations") or []):
                    quote = str(cit.get("quote") or "").strip()
                    if quote and len(quote) > 15:
                        positives.append({
                            "query": question,
                            "text_ar": quote,
                            "label": 1
                        })
                        all_texts.append(quote)
                
                # Extract from claims with evidence spans
                for claim in (r.get("claims") or []):
                    evidence = claim.get("evidence", {})
                    spans = evidence.get("supporting_spans", [])
                    for span in spans:
                        quote = str(span.get("quote") or "").strip()
                        if quote and len(quote) > 15:
                            positives.append({
                                "query": question,
                                "text_ar": quote,
                                "label": 1
                            })
                            all_texts.append(quote)
    
    # Deduplicate positives
    seen = set()
    unique_positives = []
    for p in positives:
        key = (p["query"], p["text_ar"])
        if key not in seen:
            seen.add(key)
            unique_positives.append(p)
    
    print(f"Extracted {len(unique_positives)} unique positive pairs")
    print(f"Total text pool for negatives: {len(all_texts)}")
    
    # Generate negatives (random text from different questions)
    random.seed(42)  # Deterministic
    
    n_written = 0
    with open(out_path, "w", encoding="utf-8") as out:
        for p in unique_positives:
            # Write positive
            out.write(json.dumps(p, ensure_ascii=False) + "\n")
            n_written += 1
            
            # Generate negatives using texts from other questions
            neg_candidates = [t for t in all_texts if t != p["text_ar"]]
            if neg_candidates:
                negs = random.sample(neg_candidates, min(neg_per_pos, len(neg_candidates)))
                for neg_text in negs:
                    neg = {
                        "query": p["query"],
                        "text_ar": neg_text,
                        "label": 0
                    }
                    out.write(json.dumps(neg, ensure_ascii=False) + "\n")
                    n_written += 1
    
    return n_written


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True, help="Paths to eval output JSONL files")
    parser.add_argument("--out", required=True, help="Output path for training data")
    parser.add_argument("--neg-per-pos", type=int, default=2, help="Negatives per positive")
    args = parser.parse_args()
    
    input_paths = [Path(p) for p in args.inputs]
    n = extract_training_pairs(input_paths, Path(args.out), args.neg_per_pos)
    print(f"Wrote {n} training pairs to {args.out}")


if __name__ == "__main__":
    main()
