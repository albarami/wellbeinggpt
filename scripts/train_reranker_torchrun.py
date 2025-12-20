"""Multi-GPU reranker training using torchrun and DDP.

This script trains a CrossEncoder reranker for optimal evidence ranking.
Designed for 8×A100 GPUs but works with any multi-GPU setup.

Usage (8 GPUs):
    torchrun --nproc_per_node=8 scripts/train_reranker_torchrun.py \
        --train data/reranker/train.jsonl \
        --model aubmindlab/bert-base-arabertv2 \
        --out checkpoints/reranker

Usage (single GPU):
    python scripts/train_reranker_torchrun.py \
        --train data/reranker/train.jsonl \
        --model aubmindlab/bert-base-arabertv2 \
        --out checkpoints/reranker

Training data format (JSONL):
    {"query": "...", "passage": "...", "label": 0|1}

Note: Requires torch, transformers, and sentence-transformers.
This module is optional - production system works without trained reranker.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Lazy imports for optional ML dependencies
_HAS_TORCH = False
_HAS_TRANSFORMERS = False

try:
    import torch
    from torch.utils.data import Dataset, DataLoader
    from torch.optim import AdamW
    _HAS_TORCH = True
except ImportError:
    pass

try:
    from transformers import (
        AutoTokenizer,
        AutoModelForSequenceClassification,
        get_linear_schedule_with_warmup,
    )
    _HAS_TRANSFORMERS = True
except ImportError:
    pass


@dataclass
class TrainingConfig:
    """Configuration for reranker training."""
    
    train_path: str
    model_name: str
    output_dir: str
    epochs: int = 1
    learning_rate: float = 2e-5
    batch_size: int = 8
    max_length: int = 256
    warmup_steps: int = 200
    gradient_accumulation_steps: int = 1
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    log_interval: int = 50
    save_interval: int = 500


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
                # Validate required fields (support both "passage" and "text_ar")
                if not row.get("query"):
                    continue
                if not (row.get("passage") or row.get("text_ar")):
                    continue
                if row.get("label") not in (0, 1, 0.0, 1.0):
                    continue
                self.rows.append(row)
        
        if not self.rows:
            raise ValueError(f"No valid training rows found in {path}")
    
    def __len__(self) -> int:
        return len(self.rows)
    
    def __getitem__(self, idx: int) -> tuple[str, str, int]:
        r = self.rows[idx]
        # Support both "passage" and "text_ar" field names for compatibility
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


def setup_distributed() -> tuple[int, int, bool]:
    """Setup distributed training if available."""
    if not _HAS_TORCH:
        return 0, 1, False
    
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    is_distributed = world_size > 1
    
    if is_distributed:
        torch.cuda.set_device(local_rank)
        torch.distributed.init_process_group(backend="nccl")
    
    return local_rank, world_size, is_distributed


def cleanup_distributed(is_distributed: bool) -> None:
    """Cleanup distributed training."""
    if is_distributed and torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


def train(cfg: TrainingConfig) -> None:
    """Main training loop with DDP support."""
    if not _HAS_TORCH or not _HAS_TRANSFORMERS:
        raise ImportError(
            "Training requires torch and transformers. "
            "Install with: pip install torch transformers sentence-transformers"
        )
    
    local_rank, world_size, is_distributed = setup_distributed()
    is_main = local_rank == 0
    
    if is_main:
        print(f"Training config: {cfg}")
        print(f"Distributed: {is_distributed}, World size: {world_size}")
    
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.model_name,
        num_labels=2,
    )
    
    # Move to GPU
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Wrap with DDP if distributed
    if is_distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
        )
    
    # Load dataset
    dataset = PairDataset(cfg.train_path)
    
    # Create sampler and dataloader
    if is_distributed:
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=local_rank,
            shuffle=True,
        )
    else:
        sampler = None
    
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        sampler=sampler,
        shuffle=(sampler is None),
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
    
    total_steps = cfg.epochs * len(dataloader) // cfg.gradient_accumulation_steps
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=cfg.warmup_steps,
        num_training_steps=total_steps,
    )
    
    # Training loop
    model.train()
    global_step = 0
    accumulated_loss = 0.0
    
    for epoch in range(cfg.epochs):
        if is_distributed:
            sampler.set_epoch(epoch)
        
        for step, batch in enumerate(dataloader):
            # Move batch to device
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            
            # Forward pass
            outputs = model(**batch)
            loss = outputs.loss
            
            # Scale loss for gradient accumulation
            if cfg.gradient_accumulation_steps > 1:
                loss = loss / cfg.gradient_accumulation_steps
            
            # Backward pass
            loss.backward()
            accumulated_loss += loss.item()
            
            # Optimizer step
            if (step + 1) % cfg.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1
                
                # Logging
                if is_main and global_step % cfg.log_interval == 0:
                    avg_loss = accumulated_loss / cfg.log_interval
                    lr = scheduler.get_last_lr()[0]
                    print(
                        f"epoch={epoch} step={global_step} "
                        f"loss={avg_loss:.4f} lr={lr:.2e}"
                    )
                    accumulated_loss = 0.0
                
                # Save checkpoint
                if is_main and cfg.save_interval > 0 and global_step % cfg.save_interval == 0:
                    save_checkpoint(model, tokenizer, cfg.output_dir, f"checkpoint-{global_step}")
    
    # Save final model
    if is_main:
        save_checkpoint(model, tokenizer, cfg.output_dir, "final")
        print(f"Training complete. Model saved to: {cfg.output_dir}")
    
    cleanup_distributed(is_distributed)


def save_checkpoint(
    model: Any,
    tokenizer: Any,
    output_dir: str,
    name: str,
) -> None:
    """Save model checkpoint."""
    path = Path(output_dir) / name
    path.mkdir(parents=True, exist_ok=True)
    
    # Handle DDP wrapper
    model_to_save = model.module if hasattr(model, "module") else model
    
    model_to_save.save_pretrained(str(path))
    tokenizer.save_pretrained(str(path))
    print(f"Checkpoint saved: {path}")


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Train CrossEncoder reranker with multi-GPU support"
    )
    parser.add_argument(
        "--train",
        required=True,
        help="Path to training data JSONL",
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Base model name or path (e.g., aubmindlab/bert-base-arabertv2)",
    )
    parser.add_argument(
        "--out",
        required=True,
        help="Output directory for trained model",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
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
        default=8,
        help="Batch size per GPU",
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
        default=200,
        help="Warmup steps for scheduler",
    )
    parser.add_argument(
        "--grad-accum",
        type=int,
        default=1,
        help="Gradient accumulation steps",
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=50,
        help="Log every N steps",
    )
    parser.add_argument(
        "--save-interval",
        type=int,
        default=500,
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
        gradient_accumulation_steps=args.grad_accum,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
    )
    
    train(cfg)


if __name__ == "__main__":
    main()
