"""Tests for multi-GPU reranker training script.

Tests basic functionality without requiring actual GPU/torch.
"""

import json
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock


class TestTrainingConfig:
    """Test TrainingConfig dataclass."""

    def test_config_defaults(self):
        """Test default configuration values."""
        from scripts.train_reranker_torchrun import TrainingConfig
        
        cfg = TrainingConfig(
            train_path="data/train.jsonl",
            model_name="bert-base",
            output_dir="checkpoints",
        )
        
        assert cfg.epochs == 1
        assert cfg.learning_rate == 2e-5
        assert cfg.batch_size == 8
        assert cfg.max_length == 256
        assert cfg.warmup_steps == 200

    def test_config_custom_values(self):
        """Test custom configuration values."""
        from scripts.train_reranker_torchrun import TrainingConfig
        
        cfg = TrainingConfig(
            train_path="data/train.jsonl",
            model_name="bert-base",
            output_dir="checkpoints",
            epochs=3,
            learning_rate=1e-4,
            batch_size=16,
        )
        
        assert cfg.epochs == 3
        assert cfg.learning_rate == 1e-4
        assert cfg.batch_size == 16


class TestPairDataset:
    """Test PairDataset class."""

    def test_dataset_loading(self, tmp_path: Path):
        """Test loading dataset from JSONL file."""
        from scripts.train_reranker_torchrun import PairDataset
        
        # Create temp training file (using text_ar for compatibility with prepare script)
        train_file = tmp_path / "train.jsonl"
        rows = [
            {"query": "ما هي القيم؟", "text_ar": "القيم المحورية هي...", "label": 1},
            {"query": "ما هي الركائز؟", "text_ar": "الركائز خمس هي...", "label": 1},
            {"query": "سؤال عام", "passage": "نص غير ذي صلة", "label": 0},  # Also supports "passage"
        ]
        with open(train_file, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        
        dataset = PairDataset(str(train_file))
        
        assert len(dataset) == 3
        
        # Test __getitem__
        query, passage, label = dataset[0]
        assert query == "ما هي القيم؟"
        assert passage == "القيم المحورية هي..."
        assert label == 1

    def test_dataset_filters_invalid_rows(self, tmp_path: Path):
        """Test that invalid rows are filtered out."""
        from scripts.train_reranker_torchrun import PairDataset
        
        train_file = tmp_path / "train.jsonl"
        rows = [
            {"query": "valid", "text_ar": "valid", "label": 1},  # Valid
            {"query": "", "text_ar": "valid", "label": 1},  # Invalid: empty query
            {"query": "valid", "text_ar": "", "label": 1},  # Invalid: empty passage
            {"query": "valid", "text_ar": "valid", "label": 2},  # Invalid: bad label
            {"query": "valid", "text_ar": "valid"},  # Invalid: missing label
        ]
        with open(train_file, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        
        dataset = PairDataset(str(train_file))
        
        assert len(dataset) == 1  # Only first row is valid

    def test_dataset_raises_on_empty(self, tmp_path: Path):
        """Test that empty dataset raises error."""
        from scripts.train_reranker_torchrun import PairDataset
        
        train_file = tmp_path / "train.jsonl"
        train_file.write_text("")
        
        with pytest.raises(ValueError, match="No valid training rows"):
            PairDataset(str(train_file))


class TestDistributedSetup:
    """Test distributed training setup functions."""

    def test_setup_distributed_single_gpu(self):
        """Test setup in single-GPU mode."""
        from scripts.train_reranker_torchrun import setup_distributed
        
        # Without WORLD_SIZE env var, should return single-GPU mode
        with patch.dict("os.environ", {}, clear=True):
            local_rank, world_size, is_distributed = setup_distributed()
        
        assert local_rank == 0
        assert world_size == 1
        assert is_distributed is False

    def test_cleanup_distributed_no_op(self):
        """Test cleanup when not distributed."""
        from scripts.train_reranker_torchrun import cleanup_distributed
        
        # Should not raise when not distributed
        cleanup_distributed(is_distributed=False)


class TestCollateFunction:
    """Test collate function for DataLoader."""

    def test_collate_fn_with_mock_tokenizer(self):
        """Test collate function produces correct output structure."""
        from scripts.train_reranker_torchrun import collate_fn
        
        # Skip if torch not available
        pytest.importorskip("torch")
        import torch
        
        # Create mock tokenizer
        mock_tokenizer = MagicMock()
        mock_tokenizer.return_value = {
            "input_ids": torch.tensor([[1, 2, 3], [4, 5, 6]]),
            "attention_mask": torch.tensor([[1, 1, 1], [1, 1, 0]]),
        }
        
        batch = [
            ("query1", "passage1", 1),
            ("query2", "passage2", 0),
        ]
        
        result = collate_fn(batch, mock_tokenizer, max_length=256)
        
        assert "input_ids" in result
        assert "attention_mask" in result
        assert "labels" in result
        assert result["labels"].tolist() == [1, 0]


class TestMainCLI:
    """Test CLI argument parsing."""

    def test_main_requires_args(self):
        """Test that main requires required arguments."""
        import sys
        from scripts.train_reranker_torchrun import main
        
        # Should raise SystemExit when args missing
        with patch.object(sys, "argv", ["train_reranker_torchrun.py"]):
            with pytest.raises(SystemExit):
                main()
