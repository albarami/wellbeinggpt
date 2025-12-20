"""Integration tests for eval runner output schema."""

import json
from pathlib import Path

import pytest

from eval.types import EvalOutputRow


@pytest.mark.asyncio
async def test_eval_runner_produces_schema_valid_jsonl(require_db_strict):
    """Run a tiny eval and ensure JSONL rows parse as EvalOutputRow."""
    # Use a small deterministic dataset slice.
    dataset = Path("eval/datasets/golden_slice/gold.jsonl")
    assert dataset.exists()

    # Run runner for 2 rows only.
    import subprocess
    import sys

    r = subprocess.run(
        [
            sys.executable,
            "-m",
            "eval.runner",
            "--dataset",
            str(dataset),
            "--dataset-id",
            "wellbeing",
            "--dataset-version",
            "vtest",
            "--out-dir",
            "eval/output",
            "--no-llm-only",
            "--limit",
            "2",
        ],
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert r.returncode == 0, r.stdout + "\n" + r.stderr

    run_id = (r.stdout or "").strip().splitlines()[-1]
    assert run_id

    # Validate at least one mode output exists and parses.
    out_path = Path("eval/output") / f"{run_id}__RAG_ONLY.jsonl"
    assert out_path.exists()

    rows = []
    with open(out_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(EvalOutputRow(**json.loads(line)))

    assert len(rows) == 2

    # Basic invariants
    for row in rows:
        assert row.id
        assert row.mode
        assert row.question
        assert row.answer_ar is not None
