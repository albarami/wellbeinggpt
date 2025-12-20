"""Deterministic JSONL I/O utilities for eval harness."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from eval.types import EvalOutputRow, EvalRunMetadata


@dataclass(frozen=True)
class JsonlPaths:
    output_dir: Path

    def run_jsonl_path(self, run_id: str, mode: str) -> Path:
        safe_mode = mode.replace("/", "_")
        return self.output_dir / f"{run_id}__{safe_mode}.jsonl"

    def run_meta_path(self, run_id: str) -> Path:
        return self.output_dir / f"{run_id}__meta.json"


def write_run_metadata(meta: EvalRunMetadata, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(meta.model_dump(), f, ensure_ascii=False, indent=2, sort_keys=True)


def write_jsonl_rows(rows: Iterable[EvalOutputRow], path: Path) -> int:
    """Write rows deterministically (sorted keys)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(
                json.dumps(row.model_dump(), ensure_ascii=False, sort_keys=True)
                + "\n"
            )
            n += 1
    return n


def append_jsonl_rows(rows: Iterable[EvalOutputRow], path: Path) -> int:
    """Append rows deterministically (sorted keys)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row.model_dump(), ensure_ascii=False, sort_keys=True) + "\n")
            n += 1
    return n


def read_jsonl_rows(path: Path) -> list[dict]:
    out: list[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out
