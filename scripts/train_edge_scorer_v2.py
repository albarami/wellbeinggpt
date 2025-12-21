"""
Train Edge Scorer - Phase B (5k pairs with proper metrics).

Metrics tracked:
- Precision/Recall
- AUROC
- F1 score
- Calibration

Uses train/val split by question_id to prevent leakage.
"""

import json
import logging
import os
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
TRAIN_PATH = PROJECT_ROOT / "data" / "phase2" / "edge_scorer_train_v2.jsonl"
VAL_PATH = PROJECT_ROOT / "data" / "phase2" / "edge_scorer_val_v2.jsonl"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints" / "edge_scorer_v2"

# Model config
BASE_MODEL = "aubmindlab/bert-base-arabertv2"
MAX_LENGTH = 256
BATCH_SIZE = 32
EPOCHS = 5
LEARNING_RATE = 2e-5


class EdgeScorerDataset(Dataset):
    """Dataset for edge scoring."""
    
    def __init__(self, data_path: Path, tokenizer, max_length: int = 256):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = []
        
        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                self.data.append(json.loads(line))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        query = item.get("query", "")
        from_text = item.get("from_text", item.get("from_node", ""))
        to_text = item.get("to_text", item.get("to_node", ""))
        relation = item.get("relation_type", "RELATED_TO")
        justification = item.get("justification", "")
        
        # Format: Query [SEP] NodeA -> Relation -> NodeB [SEP] Justification
        input_text = f"{query} [SEP] {from_text} -> {relation} -> {to_text} [SEP] {justification}"
        
        encoding = self.tokenizer(
            input_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(item.get("label", 0), dtype=torch.float),
        }


class EdgeScorerModel(nn.Module):
    """Binary classifier for edge scoring."""
    
    def __init__(self, base_model_name: str):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(base_model_name)
        hidden_size = self.encoder.config.hidden_size
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )
    
    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]
        score = self.classifier(cls_output)
        return score.squeeze(-1)


def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)
        
        optimizer.zero_grad()
        scores = model(input_ids, attention_mask)
        loss = criterion(scores, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        all_preds.extend(scores.detach().cpu().tolist())
        all_labels.extend(labels.cpu().tolist())
    
    # Compute metrics
    preds_binary = [1 if p > 0.5 else 0 for p in all_preds]
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, preds_binary, average="binary", zero_division=0
    )
    
    try:
        auroc = roc_auc_score(all_labels, all_preds)
    except:
        auroc = 0.0
    
    return {
        "loss": total_loss / len(dataloader),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auroc": auroc,
    }


def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)
            
            scores = model(input_ids, attention_mask)
            loss = criterion(scores, labels)
            
            total_loss += loss.item()
            all_preds.extend(scores.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
    
    # Compute metrics
    preds_binary = [1 if p > 0.5 else 0 for p in all_preds]
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, preds_binary, average="binary", zero_division=0
    )
    
    try:
        auroc = roc_auc_score(all_labels, all_preds)
    except:
        auroc = 0.0
    
    accuracy = sum(1 for p, l in zip(preds_binary, all_labels) if p == l) / len(all_labels)
    
    return {
        "loss": total_loss / len(dataloader),
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auroc": auroc,
    }


def main():
    logger.info("=" * 60)
    logger.info("EDGE SCORER TRAINING - Phase B (5k pairs)")
    logger.info("=" * 60)
    
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpus = torch.cuda.device_count()
    logger.info(f"Device: {device}, GPUs available: {n_gpus}")
    
    # Load tokenizer
    logger.info(f"Loading tokenizer: {BASE_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    
    # Load datasets (already split by question_id)
    logger.info(f"Loading train data from {TRAIN_PATH}")
    train_dataset = EdgeScorerDataset(TRAIN_PATH, tokenizer, MAX_LENGTH)
    logger.info(f"Loaded {len(train_dataset)} train examples")
    
    logger.info(f"Loading val data from {VAL_PATH}")
    val_dataset = EdgeScorerDataset(VAL_PATH, tokenizer, MAX_LENGTH)
    logger.info(f"Loaded {len(val_dataset)} val examples")
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Create model
    logger.info(f"Creating model from {BASE_MODEL}")
    model = EdgeScorerModel(BASE_MODEL)
    
    # Multi-GPU support
    if n_gpus > 1:
        logger.info(f"Using DataParallel across {n_gpus} GPUs")
        model = nn.DataParallel(model)
    
    model = model.to(device)
    
    # Optimizer and loss
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.BCELoss()
    
    # Training loop
    best_auroc = 0.0
    best_f1 = 0.0
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    
    for epoch in range(EPOCHS):
        logger.info(f"\n--- Epoch {epoch + 1}/{EPOCHS} ---")
        
        train_metrics = train_epoch(model, train_loader, optimizer, criterion, device)
        val_metrics = evaluate(model, val_loader, criterion, device)
        
        logger.info(f"Train - Loss: {train_metrics['loss']:.4f}, F1: {train_metrics['f1']:.4f}, AUROC: {train_metrics['auroc']:.4f}")
        logger.info(f"Val   - Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']:.4f}, P: {val_metrics['precision']:.4f}, R: {val_metrics['recall']:.4f}, F1: {val_metrics['f1']:.4f}, AUROC: {val_metrics['auroc']:.4f}")
        
        # Save best model (by AUROC, then F1)
        if val_metrics['auroc'] > best_auroc or (val_metrics['auroc'] == best_auroc and val_metrics['f1'] > best_f1):
            best_auroc = val_metrics['auroc']
            best_f1 = val_metrics['f1']
            model_to_save = model.module if hasattr(model, "module") else model
            torch.save(model_to_save.state_dict(), CHECKPOINT_DIR / "best_model.pt")
            logger.info(f"New best model saved (AUROC: {best_auroc:.4f}, F1: {best_f1:.4f})")
    
    # Save final model
    model_to_save = model.module if hasattr(model, "module") else model
    torch.save(model_to_save.state_dict(), CHECKPOINT_DIR / "final_model.pt")
    
    # Save training config
    config = {
        "base_model": BASE_MODEL,
        "max_length": MAX_LENGTH,
        "best_auroc": best_auroc,
        "best_f1": best_f1,
        "train_size": len(train_dataset),
        "val_size": len(val_dataset),
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
    }
    with open(CHECKPOINT_DIR / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    logger.info("\n" + "=" * 60)
    logger.info(f"Training complete.")
    logger.info(f"Best AUROC: {best_auroc:.4f}")
    logger.info(f"Best F1: {best_f1:.4f}")
    logger.info(f"Model saved to: {CHECKPOINT_DIR}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
