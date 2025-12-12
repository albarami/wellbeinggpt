"""
One-shot local ingestion utility.

Runs:
DOCX -> deterministic extraction -> canonical JSON + chunks JSONL -> Postgres load -> edges -> embeddings

Usage (PowerShell):
  $env:DATABASE_URL="postgresql+asyncpg://postgres:1234@localhost:5432/wellbeing_db"
  $env:VECTOR_BACKEND="disabled"  # or azure_search when configured
  python scripts/ingest_local_docx.py docs/source/framework_2025-10_v1.docx
"""

from __future__ import annotations

import os
import sys
import asyncio
from pathlib import Path

# Defaults for local runs; can be overridden via env BEFORE starting Python.
# Important: must be set before importing apps.api.core.database (engine is created at import time).
os.environ.setdefault(
    "DATABASE_URL",
    "postgresql+asyncpg://postgres:1234@127.0.0.1:5432/wellbeing_db",
)
os.environ.setdefault("VECTOR_BACKEND", "disabled")  # or azure_search when configured

from apps.api.ingest.docx_reader import DocxReader
from apps.api.ingest.rule_extractor import RuleExtractor
from apps.api.ingest.validator import validate_extraction
from apps.api.ingest.canonical_json import extraction_to_canonical_json, save_canonical_json
from apps.api.ingest.pipeline_framework import _expand_evidence_in_canonical
from apps.api.ingest.chunker import Chunker
from apps.api.core.database import get_session
from apps.api.ingest.loader import load_canonical_json_to_db


async def ingest_docx(path: Path) -> dict:
    data = path.read_bytes()
    parsed = DocxReader().read_bytes(data, path.name)
    extracted = RuleExtractor(framework_version="2025-10").extract(parsed)
    validation = validate_extraction(extracted, strict=True)
    if not validation.is_valid:
        msgs = [i.message for i in validation.issues][:25]
        raise SystemExit(f"Extraction validation failed (showing up to 25): {msgs}")

    canonical = _expand_evidence_in_canonical(extraction_to_canonical_json(extracted))

    derived_dir = Path("data/derived")
    derived_dir.mkdir(parents=True, exist_ok=True)
    base = path.stem
    canonical_path = derived_dir / f"{base}.canonical.json"
    chunks_path = derived_dir / f"{base}.chunks.jsonl"

    canonical.setdefault("meta", {})
    canonical["meta"]["canonical_path"] = str(canonical_path)
    canonical["meta"]["chunks_path"] = str(chunks_path)

    save_canonical_json(canonical, canonical_path)
    chunks = Chunker().chunk_canonical_json(canonical)
    Chunker().save_chunks_jsonl(chunks, str(chunks_path))

    async with get_session() as session:
        summary = await load_canonical_json_to_db(session, canonical, path.name)
        await session.commit()
    return summary


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("Usage: python scripts/ingest_local_docx.py <path-to-docx>")

    docx_path = Path(sys.argv[1])
    if not docx_path.exists():
        raise SystemExit(f"File not found: {docx_path}")

    summary = asyncio.run(ingest_docx(docx_path))
    print("INGEST SUMMARY:", summary)


if __name__ == "__main__":
    main()


