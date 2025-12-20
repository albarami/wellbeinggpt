"""
Re-ingest Framework DOCX (pipeline-consistent)

Runs the same ingestion pipeline used by tests:
DOCX -> (OCR augment if enabled) -> canonical JSON + chunks JSONL -> Postgres load (purge+upsert) -> edges.

Usage:
  python scripts/reingest_framework.py docs/source/framework_2025-10_v1.docx

Requires:
  DATABASE_URL
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from apps.api.core.database import get_session
from apps.api.ingest.loader import load_canonical_json_to_db
from apps.api.ingest.pipeline_framework import ingest_framework_docx


def _require_env(name: str) -> str:
    v = (os.getenv(name) or "").strip()
    if not v:
        raise RuntimeError(f"Missing required env var: {name}")
    return v


async def _run(docx_path: Path) -> dict[str, Any]:
    derived = Path("data/derived")
    derived.mkdir(parents=True, exist_ok=True)
    canon_path = derived / "reingest.canonical.json"
    chunks_path = derived / "reingest.chunks.jsonl"

    # Ensure supplemental OCR paragraphs are included deterministically.
    os.environ.setdefault("INGEST_OCR_FROM_IMAGES", "off")

    # pipeline function may call asyncio.run() internally, so run it off-loop
    await asyncio.to_thread(ingest_framework_docx, docx_path, canon_path, chunks_path)

    canonical = json.loads(canon_path.read_text(encoding="utf-8"))
    canonical.setdefault("meta", {})
    canonical["meta"]["canonical_path"] = str(canon_path)
    canonical["meta"]["chunks_path"] = str(chunks_path)

    async with get_session() as session:
        summary = await load_canonical_json_to_db(session, canonical, docx_path.name)
        await session.commit()
    return summary


def main(argv: list[str]) -> None:
    load_dotenv()
    _require_env("DATABASE_URL")

    if len(argv) != 2:
        raise SystemExit("Usage: python scripts/reingest_framework.py <path-to-docx>")

    docx_path = Path(argv[1])
    if not docx_path.exists():
        raise SystemExit(f"File not found: {docx_path}")

    summary = asyncio.run(_run(docx_path))
    print("REINGEST SUMMARY:", summary)


if __name__ == "__main__":
    main(sys.argv)





