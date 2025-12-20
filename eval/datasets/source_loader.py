"""Dataset source loading.

We prefer reading from the live DB when available.
If DB connection is unavailable, we fall back to deterministic ingestion artifacts:
- canonical JSON
- chunks JSONL

This ensures dataset generation is reproducible even in environments without DB.
"""

from __future__ import annotations

import asyncio
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from sqlalchemy import text

from apps.api.core.database import get_session
from apps.api.ingest.pipeline_framework import ingest_framework_docx


@dataclass(frozen=True)
class CorpusArtifacts:
    canonical_json: dict[str, Any]
    chunks_rows: list[dict[str, Any]]


def load_dotenv_if_present(dotenv_path: Path = Path(".env")) -> None:
    """Best-effort .env loader (no overrides)."""
    if not dotenv_path.exists():
        return
    try:
        for line in dotenv_path.read_text(encoding="utf-8").splitlines():
            s = line.strip()
            if not s or s.startswith("#") or "=" not in s:
                continue
            k, _, v = s.partition("=")
            k = k.strip()
            v = v.strip()
            if k and (k not in os.environ or not (os.environ.get(k) or "").strip()):
                os.environ[k] = v
    except Exception:
        return


async def _probe_db() -> bool:
    try:
        async with get_session() as session:
            _ = (
                await session.execute(text("SELECT 1"))
            ).fetchone()
        return True
    except Exception:
        return False


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


async def _ensure_local_artifacts(docx_path: Path, out_dir: Path) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    canon_path = out_dir / "eval_framework.canonical.json"
    chunks_path = out_dir / "eval_framework.chunks.jsonl"

    if canon_path.exists() and chunks_path.exists():
        return canon_path, chunks_path

    # Run ingestion in a worker thread.
    # Reason: ingest_framework_docx uses asyncio.run internally and must not be called
    # inside an already-running event loop.
    await asyncio.to_thread(ingest_framework_docx, docx_path, canon_path, chunks_path)
    return canon_path, chunks_path


async def load_corpus_artifacts(
    *,
    docx_path: Path = Path("docs/source/framework_2025-10_v1.docx"),
    out_dir: Path = Path("data/derived"),
) -> CorpusArtifacts:
    """Load corpus artifacts from DB if possible, else from local ingestion artifacts."""
    load_dotenv_if_present()

    if await _probe_db():
        async with get_session() as session:
            # Canonical hierarchy from DB.
            pillars = (
                await session.execute(text("SELECT id, name_ar FROM pillar ORDER BY id"))
            ).fetchall()
            core_values = (
                await session.execute(
                    text("SELECT id, pillar_id, name_ar FROM core_value ORDER BY id")
                )
            ).fetchall()
            sub_values = (
                await session.execute(
                    text("SELECT id, core_value_id, name_ar FROM sub_value ORDER BY id")
                )
            ).fetchall()

            canonical = {
                "pillars": [
                    {
                        "id": str(p.id),
                        "name_ar": p.name_ar,
                        "core_values": [],
                    }
                    for p in pillars
                ]
            }
            pillar_map: dict[str, dict[str, Any]] = {p["id"]: p for p in canonical["pillars"]}
            cv_map: dict[str, dict[str, Any]] = {}

            for cv in core_values:
                cv_obj = {
                    "id": str(cv.id),
                    "name_ar": cv.name_ar,
                    "sub_values": [],
                }
                cv_map[cv_obj["id"]] = cv_obj
                if str(cv.pillar_id) in pillar_map:
                    pillar_map[str(cv.pillar_id)]["core_values"].append(cv_obj)

            for sv in sub_values:
                sv_obj = {"id": str(sv.id), "name_ar": sv.name_ar}
                if str(sv.core_value_id) in cv_map:
                    cv_map[str(sv.core_value_id)]["sub_values"].append(sv_obj)

            chunks = (
                await session.execute(
                    text(
                        """
                        SELECT chunk_id, entity_type, entity_id, chunk_type, text_ar, source_anchor
                        FROM chunk
                        WHERE text_ar IS NOT NULL AND text_ar <> ''
                        ORDER BY chunk_id
                        """
                    )
                )
            ).mappings().all()
            return CorpusArtifacts(canonical_json=canonical, chunks_rows=[dict(r) for r in chunks])

    # Fallback: local artifacts
    if not docx_path.exists():
        raise RuntimeError(f"DOCX not found at {docx_path}")

    canon_path, chunks_path = await _ensure_local_artifacts(docx_path, out_dir)
    canonical = json.loads(canon_path.read_text(encoding="utf-8"))
    chunks = _read_jsonl(chunks_path)
    return CorpusArtifacts(canonical_json=canonical, chunks_rows=chunks)
