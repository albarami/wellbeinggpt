"""Optional learned reranker for retrieval.

Design goals:
- Safe default: disabled unless explicitly configured.
- Deterministic inference.
- No dependency impact for unit tests unless enabled by env.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Protocol


class Reranker(Protocol):
    def is_enabled(self) -> bool: ...

    def score(self, query: str, doc_text: str) -> float: ...


@dataclass(frozen=True)
class NullReranker:
    def is_enabled(self) -> bool:
        return False

    def score(self, query: str, doc_text: str) -> float:
        return 0.0


class CrossEncoderReranker:
    """
    Cross-encoder reranker using trained classification model.

    Enable via:
    - RERANKER_ENABLED=true
    - RERANKER_MODEL_PATH=<local path or HF model name>
    
    Supports both:
    - Classification models (2 labels) - uses softmax probability of class 1
    - Regression models (1 output) - uses raw score
    """

    def __init__(self, model_path: str):
        self._model_path = model_path
        self._model = None
        self._tokenizer = None
        self._device = None

    def is_enabled(self) -> bool:
        return True

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        
        # Lazy import to keep default installs lightweight.
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        
        self._tokenizer = AutoTokenizer.from_pretrained(self._model_path)
        self._model = AutoModelForSequenceClassification.from_pretrained(self._model_path)
        self._model.eval()
        
        # Use GPU if available
        self._device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self._model = self._model.to(self._device)

    def score(self, query: str, doc_text: str) -> float:
        self._ensure_loaded()
        q = (query or "").strip()
        d = (doc_text or "").strip()
        if not (q and d):
            return 0.0
        
        import torch
        
        inputs = self._tokenizer(
            q,
            d,
            truncation=True,
            padding=True,
            max_length=256,
            return_tensors="pt",
        )
        inputs = {k: v.to(self._device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self._model(**inputs)
            logits = outputs.logits
            
            # Handle both classification (2 labels) and regression (1 label)
            if logits.shape[-1] == 2:
                # Classification: probability of class 1 (relevant)
                probs = torch.softmax(logits, dim=-1)
                return float(probs[0, 1].item())
            else:
                # Regression: raw score
                return float(logits[0, 0].item())


def create_reranker_from_env() -> Reranker:
    enabled = os.getenv("RERANKER_ENABLED", "false").lower() in {"1", "true", "yes"}
    if not enabled:
        return NullReranker()
    model_path = (os.getenv("RERANKER_MODEL_PATH") or "").strip()
    if not model_path:
        return NullReranker()
    try:
        return CrossEncoderReranker(model_path=model_path)
    except Exception:
        return NullReranker()

