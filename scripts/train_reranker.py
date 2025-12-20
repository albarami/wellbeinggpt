"""Train a cross-encoder reranker (optional; requires ML deps).

This repo keeps training optional; production stays safe without it.

Training data format: JSONL lines with:
  {"query": "...", "text_ar": "...", "label": 1|0}

Usage:
  python -m scripts.train_reranker --train data/reranker/train_pairs.jsonl --out data/reranker/model --base-model <model>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--train", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--base-model", required=True, help="Local path or HF model name")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=16)
    args = p.parse_args()

    train_path = Path(args.train)
    rows = [json.loads(l) for l in train_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    if not rows:
        raise SystemExit("No training rows found.")

    # Lazy imports so unit tests don't require ML deps.
    from sentence_transformers import CrossEncoder, InputExample  # type: ignore
    from torch.utils.data import DataLoader  # type: ignore

    model = CrossEncoder(str(args.base_model))
    train_samples = [
        InputExample(texts=[str(r["query"]), str(r["text_ar"])], label=float(r["label"]))
        for r in rows
        if r.get("query") and r.get("text_ar") and (r.get("label") in (0, 1, 0.0, 1.0))
    ]
    if not train_samples:
        raise SystemExit("No usable training samples found.")

    train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=int(args.batch_size))
    warmup = int(0.1 * len(train_dataloader))

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    model.fit(
        train_dataloader=train_dataloader,
        epochs=int(args.epochs),
        warmup_steps=warmup,
        show_progress_bar=True,
        output_path=str(out_dir),
    )


if __name__ == "__main__":
    main()

