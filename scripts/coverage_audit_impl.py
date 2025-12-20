"""Coverage audit implementation (orchestration).

Reason: keep each script file <500 LOC.
"""

from __future__ import annotations

import asyncio
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

from sqlalchemy import text

from apps.api.core.database import get_session
from apps.api.ingest.pipeline_framework import ingest_framework_docx
from scripts.coverage_audit_entity_checks import audit_entities
from scripts.coverage_audit_utils import require_env


async def run_coverage_audit() -> dict[str, Any]:
    """Run the full coverage audit and return the report dict."""
    report: dict[str, Any] = {
        "generated_at": datetime.utcnow().isoformat(),
        "doc_hash": None,
        "doc_file": None,
        "vector_backend": os.getenv("VECTOR_BACKEND", "disabled").lower(),
        "counts": {
            "pillars_expected": 0,
            "pillars_found": 0,
            "core_values_expected": 0,
            "core_values_found": 0,
            "sub_values_expected": 0,
            "sub_values_found": 0,
            "definition_blocks_expected": 0,
            "definition_blocks_found": 0,
            "evidence_blocks_expected": 0,
            "evidence_blocks_found": 0,
            "chunks_expected": 0,
            "chunks_found": 0,
            "embeddings_expected": 0,
            "embeddings_found": 0,
        },
        "edge_counts": {},
        "missing": {"entities": [], "definitions": [], "evidence": [], "chunks": [], "embeddings": [], "graph_edges": []},
        "status": "pending",
        "notes": [],
    }

    docs_source = Path("docs/source")
    if not docs_source.exists():
        report["status"] = "error"
        report["error"] = "docs/source directory not found"
        return report

    docx_files = sorted(list(docs_source.glob("*.docx")))
    if not docx_files:
        report["status"] = "error"
        report["error"] = "No .docx files found in docs/source"
        return report

    docx_path = docx_files[0]
    report["doc_file"] = str(docx_path)

    async with get_session() as session:
        os.environ.setdefault("INGEST_OCR_FROM_IMAGES", "off")
        out_dir = Path("data/derived/coverage_audit")
        out_dir.mkdir(parents=True, exist_ok=True)
        canon_path = out_dir / "coverage_audit.canonical.json"
        chunks_path = out_dir / "coverage_audit.chunks.jsonl"

        await asyncio.to_thread(ingest_framework_docx, docx_path, canon_path, chunks_path)
        canonical = json.loads(canon_path.read_text(encoding="utf-8"))

        doc_hash = str(canonical.get("meta", {}).get("source_file_hash", "") or "").strip()
        report["doc_hash"] = doc_hash
        if not doc_hash:
            report["status"] = "error"
            report["error"] = "Canonical meta.source_file_hash missing"
            return report

        sd_row = (await session.execute(text("SELECT id FROM source_document WHERE file_hash = :h"), {"h": doc_hash})).fetchone()
        if not sd_row:
            report["missing"]["entities"].append(
                {
                    "type": "source_document",
                    "file": str(docx_path),
                    "file_hash": doc_hash,
                    "reason": "source_document not found for file_hash (did you ingest this DOCX?)",
                }
            )
            report["status"] = "incomplete"
            return report

        source_doc_id = str(sd_row.id)
        await audit_entities(session=session, report=report, canonical=canonical, source_doc_id=source_doc_id)

    all_missing_empty = all(len(v) == 0 for v in report["missing"].values())
    report["status"] = "complete" if all_missing_empty else "incomplete"
    return report


def require_db() -> None:
    require_env("DATABASE_URL")

