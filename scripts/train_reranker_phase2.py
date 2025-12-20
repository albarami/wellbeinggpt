"""Phase 2 Reranker Training on Multi-GPU (8xA100)

Trains a CrossEncoder reranker using Option B training data:
- Positives: passages actually used in PASS_FULL answers
- Hard negatives: top retrieved but not used, same-pillar wrong value

Uses DataParallel for multi-GPU training on Windows.

Usage:
    python scripts/train_reranker_phase2.py \\
        --train data/phase2/reranker_train.jsonl \\
        --model aubmindlab/bert-base-arabertv2 \\
        --out checkpoints/reranker_phase2 \\
        --epochs 3 \\
        --batch-size 64

After training:
    make eval-regression
    make bakeoff-sanity
    make bakeoff-full
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

# Add repo root
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


class RerankerDataset(Dataset):
    """Dataset for CrossEncoder reranker training."""
    
    def __init__(self, pairs: list[dict[str, Any]], tokenizer, max_length: int = 256):
        self.pairs = pairs
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        pair = self.pairs[idx]
        query = pair.get("query", "")
        passage = pair.get("text_ar", "")
        label = float(pair.get("label", 0))
        
        # Encode query + passage together for CrossEncoder
        encoding = self.tokenizer(
            query,
            passage,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(label, dtype=torch.float),
        }


class CrossEncoderModel(torch.nn.Module):
    """Simple CrossEncoder for reranking."""
    
    def __init__(self, model_name: str):
        super().__init__()
        from transformers import AutoModel
        self.encoder = AutoModel.from_pretrained(model_name)
        self.classifier = torch.nn.Linear(self.encoder.config.hidden_size, 1)
    
    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        # Use [CLS] token representation
        cls_output = outputs.last_hidden_state[:, 0, :]
        logits = self.classifier(cls_output).squeeze(-1)
        return logits


def train_epoch(model, dataloader, optimizer, criterion, device, epoch: int):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    for batch_idx, batch in enumerate(dataloader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)
        
        optimizer.zero_grad()
        logits = model(input_ids, attention_mask)
        loss = criterion(logits, labels)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        if batch_idx % 100 == 0:
            print(f"  Epoch {epoch} Batch {batch_idx}/{len(dataloader)} Loss: {loss.item():.4f}")
    
    return total_loss / num_batches if num_batches > 0 else 0.0


def evaluate(model, dataloader, criterion, device):
    """Evaluate model accuracy."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)
            
            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)
            
            total_loss += loss.item()
            
            # Accuracy (threshold at 0.5)
            preds = (torch.sigmoid(logits) > 0.5).float()
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    
    avg_loss = total_loss / len(dataloader) if len(dataloader) > 0 else 0.0
    accuracy = correct / total if total > 0 else 0.0
    return avg_loss, accuracy


def main():
    parser = argparse.ArgumentParser(description="Train Phase 2 reranker")
    parser.add_argument("--train", type=str, required=True, help="Training data JSONL path")
    parser.add_argument("--model", type=str, default="aubmindlab/bert-base-arabertv2", help="Base model")
    parser.add_argument("--out", type=str, default="checkpoints/reranker_phase2", help="Output directory")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size per GPU")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--max-length", type=int, default=256, help="Max sequence length")
    parser.add_argument("--val-split", type=float, default=0.1, help="Validation split")
    args = parser.parse_args()
    
    print("=" * 60)
    print("PHASE 2 RERANKER TRAINING")
    print("=" * 60)
    
    # Check GPU availability
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        print(f"CUDA available: {num_gpus} GPU(s)")
        for i in range(num_gpus):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
        device = torch.device("cuda")
    else:
        print("WARNING: CUDA not available, using CPU")
        device = torch.device("cpu")
        num_gpus = 0
    
    # Load training data
    print(f"\nLoading training data from {args.train}...")
    train_path = Path(args.train)
    if not train_path.exists():
        print(f"ERROR: Training file not found: {train_path}")
        return 1
    
    pairs = []
    with open(train_path, "r", encoding="utf-8") as f:
        for line in f:
            pairs.append(json.loads(line.strip()))
    
    print(f"  Loaded {len(pairs)} training pairs")
    
    # Check label distribution
    pos_count = sum(1 for p in pairs if p.get("label") == 1)
    neg_count = len(pairs) - pos_count
    print(f"  Positives: {pos_count}, Negatives: {neg_count}")
    
    # Shuffle and split
    random.shuffle(pairs)
    val_size = int(len(pairs) * args.val_split)
    val_pairs = pairs[:val_size]
    train_pairs = pairs[val_size:]
    print(f"  Train: {len(train_pairs)}, Validation: {len(val_pairs)}")
    
    # Load tokenizer
    print(f"\nLoading tokenizer: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    
    # Create datasets
    train_dataset = RerankerDataset(train_pairs, tokenizer, max_length=args.max_length)
    val_dataset = RerankerDataset(val_pairs, tokenizer, max_length=args.max_length)
    
    # Adjust batch size for multi-GPU
    effective_batch_size = args.batch_size * max(1, num_gpus)
    print(f"Effective batch size: {effective_batch_size} (per-GPU: {args.batch_size})")
    
    train_loader = DataLoader(train_dataset, batch_size=effective_batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=effective_batch_size, shuffle=False, num_workers=0)
    
    # Create model
    print(f"\nLoading model: {args.model}")
    model = CrossEncoderModel(args.model)
    
    # Multi-GPU with DataParallel
    if num_gpus > 1:
        print(f"Using DataParallel across {num_gpus} GPUs")
        model = torch.nn.DataParallel(model)
    
    model = model.to(device)
    
    # Optimizer and loss
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    criterion = torch.nn.BCEWithLogitsLoss()
    
    # Training loop
    print(f"\nStarting training for {args.epochs} epochs...")
    best_val_acc = 0.0
    
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device, epoch)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        epoch_time = time.time() - t0
        
        print(f"Epoch {epoch}/{args.epochs}: "
              f"Train Loss={train_loss:.4f}, "
              f"Val Loss={val_loss:.4f}, "
              f"Val Acc={val_acc:.4f}, "
              f"Time={epoch_time:.1f}s")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            out_dir = Path(args.out)
            out_dir.mkdir(parents=True, exist_ok=True)
            
            # Save model state
            model_to_save = model.module if hasattr(model, "module") else model
            torch.save({
                "epoch": epoch,
                "model_state_dict": model_to_save.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc": val_acc,
                "val_loss": val_loss,
            }, out_dir / "best_model.pt")
            
            # Save tokenizer
            tokenizer.save_pretrained(out_dir / "tokenizer")
            
            print(f"  -> Saved best model (acc={val_acc:.4f})")
    
    # Save final model
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    model_to_save = model.module if hasattr(model, "module") else model
    torch.save({
        "epoch": args.epochs,
        "model_state_dict": model_to_save.state_dict(),
        "val_acc": val_acc,
    }, out_dir / "final_model.pt")
    
    # Save training config
    config = {
        "base_model": args.model,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "max_length": args.max_length,
        "train_pairs": len(train_pairs),
        "val_pairs": len(val_pairs),
        "best_val_acc": best_val_acc,
        "num_gpus": num_gpus,
    }
    with open(out_dir / "training_config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    
    print()
    print("=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Best validation accuracy: {best_val_acc:.4f}")
    print(f"Model saved to: {out_dir}")
    print()
    print("Next steps:")
    print("  1. make eval-regression")
    print("  2. make bakeoff-sanity")
    print("  3. make bakeoff-full")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
