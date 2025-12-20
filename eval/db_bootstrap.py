"""Evaluation DB bootstrap.

Ensures the database schema exists and that the framework DOCX has been ingested
into Postgres before running evaluation.

This mirrors the best-effort behavior used in tests/conftest.py.
"""

from __future__ import annotations

import asyncio
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from apps.api.core.database import get_session
from apps.api.core.schema_bootstrap import bootstrap_db
from apps.api.ingest.pipeline_framework import ingest_framework_docx
from apps.api.ingest.loader import load_canonical_json_to_db
from apps.api.ingest.chunk_span_store import populate_chunk_spans_for_source
from apps.api.ingest.scholar_notes_loader import ingest_scholar_notes_jsonl


@dataclass(frozen=True)
class DbBootstrapConfig:
    docx_path: Path = Path("docs/source/framework_2025-10_v1.docx")
    out_dir: Path = Path("data/derived")

    # By default, ingestion disables OCR during eval for determinism.
    ingest_ocr_from_images: str = "off"


def _require_env(name: str) -> str:
    v = (os.getenv(name) or "").strip()
    if not v:
        raise RuntimeError(f"Missing required env var: {name}")
    return v


def _is_valid_jsonl(path: Path, *, max_errors: int = 0) -> bool:
    """
    Validate that a JSONL file can be fully parsed.

    Reason: interrupted runs can leave partially-written JSONL which then causes
    JSONDecodeError during DB bootstrap.
    """
    if not path.exists():
        return False
    errors = 0
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            s = line.strip()
            if not s:
                continue
            try:
                json.loads(s)
            except Exception:
                errors += 1
                if errors > max_errors:
                    return False
    return True


def _is_valid_json(path: Path) -> bool:
    """
    Validate that a JSON file can be fully parsed.

    Reason: interrupted runs can leave partially-written JSON which then causes
    JSONDecodeError during DB bootstrap.
    """
    if not path.exists():
        return False
    try:
        json.loads(path.read_text(encoding="utf-8"))
        return True
    except Exception:
        return False


def _safe_unlink(path: Path) -> None:
    try:
        if path.exists():
            path.unlink()
    except Exception:
        return


async def ensure_db_populated(cfg: DbBootstrapConfig) -> None:
    """Ensure schema is applied and framework is loaded."""
    _require_env("DATABASE_URL")

    # Apply schema (idempotent best-effort).
    await bootstrap_db()

    if not cfg.docx_path.exists():
        raise RuntimeError(f"Framework DOCX not found: {cfg.docx_path}")

    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    canon_path = cfg.out_dir / "eval_runner.canonical.json"
    chunks_path = cfg.out_dir / "eval_runner.chunks.jsonl"

    # If artifacts are corrupted (partial write), delete and regenerate.
    # This makes reruns self-healing and prevents JSONDecodeError crashes.
    if (chunks_path.exists() and (not _is_valid_jsonl(chunks_path))) or (
        canon_path.exists() and (not _is_valid_json(canon_path))
    ):
        _safe_unlink(chunks_path)
        _safe_unlink(canon_path)

    # Ingest DOCX to canonical+chunks in a thread (ingest uses asyncio.run internally).
    os.environ.setdefault("INGEST_OCR_FROM_IMAGES", cfg.ingest_ocr_from_images)
    await asyncio.to_thread(ingest_framework_docx, cfg.docx_path, canon_path, chunks_path)

    canonical = json.loads(canon_path.read_text(encoding="utf-8"))

    async with get_session() as session:
        summary = await load_canonical_json_to_db(session, canonical, cfg.docx_path.name)
        # Persist sentence spans for stable citation offsets (best-effort).
        try:
            source_doc_id = str(summary.get("source_doc_id") or "")
            if source_doc_id:
                await populate_chunk_spans_for_source(session, source_doc_id)
        except Exception:
            pass

        # Commit framework ingestion before generating scholar notes.
        # Reason: note generation opens a new session and must see chunks.
        try:
            await session.commit()
        except Exception:
            pass

        # Ingest scholar notes pack for depth (required if present).
        notes_path = Path("data/scholar_notes/notes_v1.jsonl")
        if notes_path.exists() and notes_path.stat().st_size > 0:
            # Regenerate a deterministic starter pack from the current DB so chunk_ids match.
            from scripts.generate_scholar_notes_v1 import _run as _gen_notes  # type: ignore

            await _gen_notes(out_path=notes_path, limit=12, version="v1")
            await ingest_scholar_notes_jsonl(
                session=session,
                notes_jsonl_path=str(notes_path),
                pack_name="scholar_notes_v1",
            )
        try:
            await session.commit()
        except Exception:
            pass
