"""Dataset JSONL I/O."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from eval.datasets.types import DatasetRow


def read_dataset_jsonl(path: Path) -> list[DatasetRow]:
    out: list[DatasetRow] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(DatasetRow(**json.loads(line)))
    return out


def write_dataset_jsonl(rows: Iterable[DatasetRow], path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r.model_dump(), ensure_ascii=False, sort_keys=True) + "\n")
            n += 1
    return n
