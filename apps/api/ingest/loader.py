"""Database Loader Module.

This module is intentionally kept small (<500 LOC). The implementation was
split into focused modules:
- `loader_meta.py` (source_document + ingestion_run)
- `loader_entities.py` (pillar/core/sub upserts)
- `loader_text.py` (text_block + evidence)
- `loader_chunks.py` (chunk/chunk_ref + embeddings)
- `loader_edges.py` (graph edges)
- `loader_pipeline.py` (canonical JSON orchestration)

All public functions remain import-compatible from `apps.api.ingest.loader`.
"""

from __future__ import annotations

from apps.api.ingest.loader_chunks import embed_all_chunks_for_source, load_chunks_jsonl
from apps.api.ingest.loader_edges import build_edges_for_source
from apps.api.ingest.loader_entities import load_core_value, load_pillar, load_sub_value
from apps.api.ingest.loader_meta import (
    _scalar_one_or_none,
    complete_ingestion_run,
    create_ingestion_run,
    create_source_document,
)
from apps.api.ingest.loader_pipeline import load_canonical_json_to_db
from apps.api.ingest.loader_text import load_evidence, load_text_block

__all__ = [
    "_scalar_one_or_none",
    "create_source_document",
    "create_ingestion_run",
    "complete_ingestion_run",
    "load_pillar",
    "load_core_value",
    "load_sub_value",
    "load_text_block",
    "load_evidence",
    "load_canonical_json_to_db",
    "load_chunks_jsonl",
    "embed_all_chunks_for_source",
    "build_edges_for_source",
]
