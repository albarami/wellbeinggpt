"""
Train Edge Scorer - Binary classifier for edge strength/relevance.

This is the Phase A prototype (212 pairs). It validates the training pipeline
before scaling to 2k-10k pairs.

Input per example:
- nodeA text (resolved from entity ID)
- nodeB text (resolved from entity ID)
- relation_type
- justification span text
- query text

Output:
- score ∈ [0,1] (edge strength / accept vs reject)
"""

import json
import logging
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import AutoTokenizer, AutoModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "phase2" / "edge_scorer_train.jsonl"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints" / "edge_scorer"
ENTITY_CACHE_PATH = PROJECT_ROOT / "data" / "phase2" / "entity_labels.json"

# Model config
BASE_MODEL = "aubmindlab/bert-base-arabertv2"
MAX_LENGTH = 256
BATCH_SIZE = 16
EPOCHS = 5
LEARNING_RATE = 2e-5


class EdgeScorerDataset(Dataset):
    """Dataset for edge scoring."""
    
    def __init__(self, data: list[dict], tokenizer, entity_labels: dict, max_length: int = 256):
        self.data = data
        self.tokenizer = tokenizer
        self.entity_labels = entity_labels
        self.max_length = max_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Resolve entity labels
        from_entity = item.get("from_entity", "")
        to_entity = item.get("to_entity", "")
        nodeA_text = self.entity_labels.get(from_entity, from_entity)
        nodeB_text = self.entity_labels.get(to_entity, to_entity)
        
        # Build input text
        relation = item.get("relation_type", "") or "RELATED_TO"
        justification = item.get("justification", "") or ""
        query = item.get("query", "") or ""
        
        # Format: [CLS] Query [SEP] NodeA -> Relation -> NodeB [SEP] Justification
        input_text = f"{query} [SEP] {nodeA_text} -> {relation} -> {nodeB_text} [SEP] {justification}"
        
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
        # Use [CLS] token representation
        cls_output = outputs.last_hidden_state[:, 0, :]
        score = self.classifier(cls_output)
        return score.squeeze(-1)


def load_entity_labels() -> dict:
    """Load or build entity label cache."""
    if ENTITY_CACHE_PATH.exists():
        with open(ENTITY_CACHE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    
    # Build from database
    logger.info("Building entity label cache from database...")
    import asyncio
    from sqlalchemy import text
    
    async def fetch_labels():
        from apps.api.core.database import get_session
        
        labels = {}
        async with get_session() as session:
            # Pillars
            result = await session.execute(text("SELECT pillar_id, label_ar FROM pillar"))
            for row in result.fetchall():
                labels[f"pillar:{row[0]}"] = row[1] or row[0]
            
            # Core values
            result = await session.execute(text("SELECT value_id, label_ar FROM core_value"))
            for row in result.fetchall():
                labels[f"core_value:{row[0]}"] = row[1] or row[0]
            
            # Sub values
            result = await session.execute(text("SELECT value_id, label_ar FROM sub_value"))
            for row in result.fetchall():
                labels[f"sub_value:{row[0]}"] = row[1] or row[0]
        
        return labels
    
    try:
        labels = asyncio.run(fetch_labels())
        # Save cache
        ENTITY_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(ENTITY_CACHE_PATH, "w", encoding="utf-8") as f:
            json.dump(labels, f, ensure_ascii=False, indent=2)
        logger.info(f"Cached {len(labels)} entity labels")
        return labels
    except Exception as e:
        logger.warning(f"Could not fetch from DB: {e}. Using IDs as labels.")
        return {}


def load_training_data() -> list[dict]:
    """Load training data from JSONL."""
    data = []
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    return data


def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
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
        predictions = (scores > 0.5).float()
        correct += (predictions == labels).sum().item()
        total += labels.size(0)
    
    return total_loss / len(dataloader), correct / total


def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)
            
            scores = model(input_ids, attention_mask)
            loss = criterion(scores, labels)
            
            total_loss += loss.item()
            predictions = (scores > 0.5).float()
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
    
    return total_loss / len(dataloader), correct / total


def main():
    logger.info("=" * 60)
    logger.info("EDGE SCORER TRAINING - Phase A (212 pairs prototype)")
    logger.info("=" * 60)
    
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpus = torch.cuda.device_count()
    logger.info(f"Device: {device}, GPUs available: {n_gpus}")
    
    # Load data
    logger.info(f"Loading training data from {DATA_PATH}")
    data = load_training_data()
    logger.info(f"Loaded {len(data)} examples")
    
    pos_count = sum(1 for d in data if d.get("label") == 1)
    neg_count = len(data) - pos_count
    logger.info(f"Positives: {pos_count}, Negatives: {neg_count}")
    
    # Load entity labels
    entity_labels = load_entity_labels()
    logger.info(f"Loaded {len(entity_labels)} entity labels")
    
    # Load tokenizer
    logger.info(f"Loading tokenizer: {BASE_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    
    # Create dataset
    dataset = EdgeScorerDataset(data, tokenizer, entity_labels, MAX_LENGTH)
    
    # Split data
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    logger.info(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    
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
    best_val_acc = 0.0
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    
    for epoch in range(EPOCHS):
        logger.info(f"\n--- Epoch {epoch + 1}/{EPOCHS} ---")
        
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        
        logger.info(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        logger.info(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            # Save best model
            model_to_save = model.module if hasattr(model, "module") else model
            torch.save(model_to_save.state_dict(), CHECKPOINT_DIR / "best_model.pt")
            logger.info(f"New best model saved (val_acc: {val_acc:.4f})")
    
    # Save final model
    model_to_save = model.module if hasattr(model, "module") else model
    torch.save(model_to_save.state_dict(), CHECKPOINT_DIR / "final_model.pt")
    
    # Save training config
    config = {
        "base_model": BASE_MODEL,
        "max_length": MAX_LENGTH,
        "best_val_acc": best_val_acc,
        "train_size": train_size,
        "val_size": val_size,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
    }
    with open(CHECKPOINT_DIR / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    logger.info("\n" + "=" * 60)
    logger.info(f"Training complete. Best val accuracy: {best_val_acc:.4f}")
    logger.info(f"Model saved to: {CHECKPOINT_DIR}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
