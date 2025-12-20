"""Multi-GPU reranker training using DataParallel for Windows compatibility.

This script trains a CrossEncoder reranker for optimal evidence ranking.
Uses DataParallel to utilize all available GPUs.

Usage:
    python scripts/train_reranker_dp.py \
        --train data/reranker/train.jsonl \
        --model aubmindlab/bert-base-arabertv2 \
        --out checkpoints/reranker \
        --epochs 3 \
        --batch-size 256

Training data format (JSONL):
    {"query": "...", "passage": "...", "label": 0|1}
    OR
    {"query": "...", "text_ar": "...", "label": 0|1}
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)


@dataclass
class TrainingConfig:
    """Configuration for reranker training."""
    
    train_path: str
    model_name: str
    output_dir: str
    epochs: int = 3
    learning_rate: float = 2e-5
    batch_size: int = 256  # Total batch size across all GPUs
    max_length: int = 256
    warmup_steps: int = 100
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    log_interval: int = 10
    save_interval: int = 100


class PairDataset(Dataset):
    """Dataset for query-passage pairs with binary relevance labels."""
    
    def __init__(self, path: str):
        self.rows: list[dict[str, Any]] = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                if not row.get("query"):
                    continue
                if not (row.get("passage") or row.get("text_ar")):
                    continue
                if row.get("label") not in (0, 1, 0.0, 1.0):
                    continue
                self.rows.append(row)
        
        if not self.rows:
            raise ValueError(f"No valid training rows found in {path}")
        
        print(f"Loaded {len(self.rows)} training pairs")
    
    def __len__(self) -> int:
        return len(self.rows)
    
    def __getitem__(self, idx: int) -> tuple[str, str, int]:
        r = self.rows[idx]
        passage = r.get("passage") or r.get("text_ar") or ""
        return str(r["query"]), str(passage), int(r["label"])


def collate_fn(batch: list[tuple[str, str, int]], tokenizer: Any, max_length: int) -> dict[str, Any]:
    """Collate function for DataLoader."""
    queries, passages, labels = zip(*batch)
    
    encodings = tokenizer(
        list(queries),
        list(passages),
        truncation=True,
        padding=True,
        max_length=max_length,
        return_tensors="pt",
    )
    encodings["labels"] = torch.tensor(labels, dtype=torch.long)
    
    return encodings


def train(cfg: TrainingConfig) -> None:
    """Main training loop with DataParallel for multi-GPU."""
    
    # Check GPU availability
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available. This script requires GPUs.")
    
    num_gpus = torch.cuda.device_count()
    print(f"=" * 60)
    print(f"MULTI-GPU TRAINING: {num_gpus} GPUs detected")
    for i in range(num_gpus):
        props = torch.cuda.get_device_properties(i)
        print(f"  GPU {i}: {props.name} ({props.total_memory / 1024**3:.1f} GB)")
    print(f"=" * 60)
    print(f"Config: {cfg}")
    print(f"=" * 60)
    
    # Load tokenizer and model
    print("Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.model_name,
        num_labels=2,
    )
    
    # Wrap with DataParallel to use all GPUs
    device = torch.device("cuda:0")
    model = model.to(device)
    
    if num_gpus > 1:
        print(f"Wrapping model with DataParallel across {num_gpus} GPUs")
        model = torch.nn.DataParallel(model)
    
    # Load dataset
    print(f"Loading dataset from {cfg.train_path}...")
    dataset = PairDataset(cfg.train_path)
    
    # Per-GPU batch size
    per_gpu_batch = cfg.batch_size // num_gpus
    actual_batch_size = per_gpu_batch * num_gpus
    print(f"Batch size: {actual_batch_size} total ({per_gpu_batch} per GPU)")
    
    dataloader = DataLoader(
        dataset,
        batch_size=actual_batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_fn(b, tokenizer, cfg.max_length),
        num_workers=0,  # Avoid multiprocessing issues on Windows
        pin_memory=True,
    )
    
    # Optimizer and scheduler
    optimizer = AdamW(
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )
    
    total_steps = cfg.epochs * len(dataloader)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=cfg.warmup_steps,
        num_training_steps=total_steps,
    )
    
    print(f"Total training steps: {total_steps}")
    print(f"=" * 60)
    print("Starting training...")
    
    # Training loop
    model.train()
    global_step = 0
    accumulated_loss = 0.0
    start_time = time.time()
    
    for epoch in range(cfg.epochs):
        epoch_start = time.time()
        epoch_loss = 0.0
        
        for step, batch in enumerate(dataloader):
            # Move batch to device
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            
            # Forward pass
            outputs = model(**batch)
            loss = outputs.loss
            
            # Handle DataParallel loss averaging
            if num_gpus > 1:
                loss = loss.mean()
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
            
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            
            global_step += 1
            accumulated_loss += loss.item()
            epoch_loss += loss.item()
            
            # Logging
            if global_step % cfg.log_interval == 0:
                avg_loss = accumulated_loss / cfg.log_interval
                lr = scheduler.get_last_lr()[0]
                elapsed = time.time() - start_time
                samples_per_sec = (global_step * actual_batch_size) / elapsed
                
                print(
                    f"epoch={epoch+1}/{cfg.epochs} "
                    f"step={global_step}/{total_steps} "
                    f"loss={avg_loss:.4f} "
                    f"lr={lr:.2e} "
                    f"speed={samples_per_sec:.1f} samples/sec"
                )
                accumulated_loss = 0.0
            
            # Save checkpoint
            if cfg.save_interval > 0 and global_step % cfg.save_interval == 0:
                save_checkpoint(model, tokenizer, cfg.output_dir, f"checkpoint-{global_step}")
        
        epoch_elapsed = time.time() - epoch_start
        avg_epoch_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch+1} completed in {epoch_elapsed:.1f}s, avg_loss={avg_epoch_loss:.4f}")
    
    # Save final model
    save_checkpoint(model, tokenizer, cfg.output_dir, "final")
    
    total_elapsed = time.time() - start_time
    print(f"=" * 60)
    print(f"Training complete in {total_elapsed:.1f}s")
    print(f"Final model saved to: {cfg.output_dir}/final")
    print(f"=" * 60)


def save_checkpoint(
    model: Any,
    tokenizer: Any,
    output_dir: str,
    name: str,
) -> None:
    """Save model checkpoint."""
    path = Path(output_dir) / name
    path.mkdir(parents=True, exist_ok=True)
    
    # Handle DataParallel wrapper
    model_to_save = model.module if hasattr(model, "module") else model
    
    model_to_save.save_pretrained(str(path))
    tokenizer.save_pretrained(str(path))
    print(f"Checkpoint saved: {path}")


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Train CrossEncoder reranker with multi-GPU DataParallel"
    )
    parser.add_argument(
        "--train",
        required=True,
        help="Path to training data JSONL",
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Base model name or path",
    )
    parser.add_argument(
        "--out",
        required=True,
        help="Output directory for trained model",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=2e-5,
        help="Learning rate",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Total batch size across all GPUs",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=256,
        help="Max sequence length",
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=100,
        help="Warmup steps for scheduler",
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=10,
        help="Log every N steps",
    )
    parser.add_argument(
        "--save-interval",
        type=int,
        default=100,
        help="Save checkpoint every N steps (0 to disable)",
    )
    
    args = parser.parse_args()
    
    cfg = TrainingConfig(
        train_path=args.train,
        model_name=args.model,
        output_dir=args.out,
        epochs=args.epochs,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        max_length=args.max_length,
        warmup_steps=args.warmup_steps,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
    )
    
    train(cfg)


if __name__ == "__main__":
    main()
