"""
Edge Scorer - Ranks/scores candidate edges for selection.

Purpose:
- Given candidate grounded edges (already evidence-backed), edge scorer ranks them
- Helps select stronger, more meaningful edges
- Improves diversity and coherence in networks/loops
- NOT a safety module - safety gates remain separate

Integration:
- Runs after candidate edge enumeration
- Runs before used_edges selection
- Only ranks already grounded candidates (never fabricates)
"""

import json
import logging
import os
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel

logger = logging.getLogger(__name__)

# Paths
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints" / "edge_scorer"


class EdgeScorerModel(nn.Module):
    """Binary classifier for edge scoring (same architecture as training)."""
    
    def __init__(self, base_model_name: str = "aubmindlab/bert-base-arabertv2"):
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


class EdgeScorer:
    """Edge scorer for ranking candidate edges."""
    
    def __init__(self, model_path: Optional[str] = None):
        self.model = None
        self.tokenizer = None
        self.device = None
        self._enabled = False
        
        if model_path is None:
            model_path = str(CHECKPOINT_DIR)
        
        self._load_model(model_path)
    
    def _load_model(self, model_path: str) -> None:
        """Load the edge scorer model."""
        model_dir = Path(model_path)
        best_model_path = model_dir / "best_model.pt"
        config_path = model_dir / "config.json"
        
        if not best_model_path.exists():
            logger.warning(f"Edge scorer model not found at {best_model_path}")
            return
        
        try:
            # Load config
            if config_path.exists():
                with open(config_path, "r") as f:
                    config = json.load(f)
                base_model = config.get("base_model", "aubmindlab/bert-base-arabertv2")
                self.max_length = config.get("max_length", 256)
            else:
                base_model = "aubmindlab/bert-base-arabertv2"
                self.max_length = 256
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(base_model)
            
            # Load model
            self.model = EdgeScorerModel(base_model)
            state_dict = torch.load(best_model_path, map_location="cpu")
            self.model.load_state_dict(state_dict)
            
            # Set device
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model = self.model.to(self.device)
            self.model.eval()
            
            self._enabled = True
            logger.info(f"Edge scorer loaded from {model_path}")
            
        except Exception as e:
            logger.error(f"Failed to load edge scorer: {e}")
            self._enabled = False
    
    def is_enabled(self) -> bool:
        """Check if edge scorer is ready."""
        return self._enabled
    
    def score_edge(
        self,
        query: str,
        from_node_text: str,
        to_node_text: str,
        relation_type: str,
        justification: str,
    ) -> float:
        """
        Score a single candidate edge.
        
        Args:
            query: The user question
            from_node_text: Label/text of source node
            to_node_text: Label/text of target node
            relation_type: Relation type (ENABLES, REINFORCES, etc.)
            justification: Justification span text
        
        Returns:
            Score in [0, 1] where higher is better
        """
        if not self._enabled:
            return 0.5  # Neutral score if not enabled
        
        try:
            # Format input (same as training)
            input_text = f"{query} [SEP] {from_node_text} -> {relation_type} -> {to_node_text} [SEP] {justification}"
            
            encoding = self.tokenizer(
                input_text,
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            
            input_ids = encoding["input_ids"].to(self.device)
            attention_mask = encoding["attention_mask"].to(self.device)
            
            with torch.no_grad():
                score = self.model(input_ids, attention_mask)
            
            return float(score.cpu().item())
            
        except Exception as e:
            logger.warning(f"Edge scoring failed: {e}")
            return 0.5
    
    def score_edges_batch(
        self,
        query: str,
        edges: list[dict],
    ) -> list[tuple[dict, float]]:
        """
        Score a batch of candidate edges.
        
        Args:
            query: The user question
            edges: List of edge dicts with keys:
                - from_node_text or from_label
                - to_node_text or to_label
                - relation_type
                - justification or justification_spans
        
        Returns:
            List of (edge, score) tuples, sorted by score descending
        """
        if not self._enabled or not edges:
            return [(e, 0.5) for e in edges]
        
        try:
            # Prepare batch
            input_texts = []
            for edge in edges:
                from_text = edge.get("from_node_text") or edge.get("from_label") or edge.get("from_node", "")
                to_text = edge.get("to_node_text") or edge.get("to_label") or edge.get("to_node", "")
                relation = edge.get("relation_type", "RELATED_TO")
                
                # Get justification text
                justification = edge.get("justification", "")
                if not justification:
                    spans = edge.get("justification_spans", [])
                    if spans and isinstance(spans, list):
                        justification = " ".join(str(s.get("quote", "")) for s in spans[:2])
                
                input_text = f"{query} [SEP] {from_text} -> {relation} -> {to_text} [SEP] {justification}"
                input_texts.append(input_text)
            
            # Tokenize batch
            encodings = self.tokenizer(
                input_texts,
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            
            input_ids = encodings["input_ids"].to(self.device)
            attention_mask = encodings["attention_mask"].to(self.device)
            
            with torch.no_grad():
                scores = self.model(input_ids, attention_mask)
            
            scores_list = scores.cpu().tolist()
            if not isinstance(scores_list, list):
                scores_list = [scores_list]
            
            # Pair edges with scores
            scored_edges = list(zip(edges, scores_list))
            
            # Sort by score descending
            scored_edges.sort(key=lambda x: x[1], reverse=True)
            
            return scored_edges
            
        except Exception as e:
            logger.warning(f"Batch edge scoring failed: {e}")
            return [(e, 0.5) for e in edges]
    
    def rank_and_select(
        self,
        query: str,
        edges: list[dict],
        top_k: Optional[int] = None,
        min_score: float = 0.3,
    ) -> list[dict]:
        """
        Rank edges and return top-k with score above threshold.
        
        Args:
            query: User question
            edges: Candidate edges
            top_k: Maximum edges to return (None = all above threshold)
            min_score: Minimum score threshold
        
        Returns:
            Selected edges with 'edge_score' added
        """
        scored = self.score_edges_batch(query, edges)
        
        # Filter by threshold
        filtered = [(e, s) for e, s in scored if s >= min_score]
        
        # Apply top_k limit
        if top_k is not None:
            filtered = filtered[:top_k]
        
        # Add score to edges
        result = []
        for edge, score in filtered:
            edge_copy = edge.copy()
            edge_copy["edge_score"] = score
            result.append(edge_copy)
        
        return result


# Module-level singleton
_edge_scorer: Optional[EdgeScorer] = None


def get_edge_scorer() -> EdgeScorer:
    """Get or create edge scorer singleton."""
    global _edge_scorer
    
    if _edge_scorer is None:
        enabled = os.getenv("EDGE_SCORER_ENABLED", "false").lower() in {"1", "true", "yes"}
        model_path = os.getenv("EDGE_SCORER_MODEL_PATH", str(CHECKPOINT_DIR))
        
        if enabled:
            _edge_scorer = EdgeScorer(model_path)
        else:
            # Return a disabled scorer
            _edge_scorer = EdgeScorer(None)
            _edge_scorer._enabled = False
    
    return _edge_scorer


def should_use_edge_scorer(intent: Optional[str] = None) -> bool:
    """
    Determine if edge scorer should be used for this request.
    
    Currently: use for cross_pillar and network intents.
    """
    scorer = get_edge_scorer()
    if not scorer.is_enabled():
        return False
    
    intent_lower = (intent or "").lower().strip()
    
    # Use edge scorer for network/cross-pillar intents
    edge_scorer_intents = {
        "cross_pillar",
        "cross_pillar_path",
        "network_build",
        "global_synthesis",
        "world_model",
        "pillar_relationship",
    }
    
    return intent_lower in edge_scorer_intents
