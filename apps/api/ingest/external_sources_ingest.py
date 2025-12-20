"""External corpus ingestion (file-based).

This ingests user-provided external sources as:
- `source_document` rows (provenance via sha256)
- `chunk` rows (evidence-only retrieval)
- optional embeddings via existing embedding pipeline

We keep the ingestion conservative:
- Supported formats: txt, md
- Chunking: deterministic paragraph/sentence grouping
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from apps.api.ingest.external_sources_manifest import ExternalSourceManifestRow, sha256_file_bytes
from apps.api.ingest.loader_chunks import embed_all_chunks_for_source
from apps.api.ingest.loader_meta import complete_ingestion_run, create_ingestion_run, create_source_document


@dataclass(frozen=True)
class ExternalIngestResult:
    source_id: str
    source_doc_id: str
    run_id: str
    chunks: int
    embeddings: int


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="strict")


def _split_paragraphs(text: str) -> list[str]:
    # Normalize newlines and split on blank lines.
    t = (text or "").replace("\r\n", "\n").replace("\r", "\n")
    paras = [p.strip() for p in re.split(r"\n\s*\n+", t) if p.strip()]
    return paras


def _chunk_text(paras: list[str], *, max_chars: int = 1200) -> list[str]:
    chunks: list[str] = []
    cur: list[str] = []
    cur_len = 0
    for p in paras:
        if not p:
            continue
        if cur_len + len(p) + 2 > max_chars and cur:
            chunks.append("\n\n".join(cur).strip())
            cur = []
            cur_len = 0
        cur.append(p)
        cur_len += len(p) + 2
    if cur:
        chunks.append("\n\n".join(cur).strip())
    return [c for c in chunks if c]


def _stable_chunk_id(*, source_id: str, idx: int, text: str) -> str:
    # Deterministic short hash (avoid long IDs; keep <=50 chars).
    import hashlib

    h = hashlib.sha256((source_id + "\n" + str(idx) + "\n" + (text or "")).encode("utf-8")).hexdigest()[:12]
    return f"CHX_{source_id[:18]}_{idx:03d}_{h}"


async def ingest_external_source(
    *,
    session: AsyncSession,
    row: ExternalSourceManifestRow,
    repo_root: Path,
    embed: bool = True,
) -> ExternalIngestResult:
    """
    Ingest one external source described by the manifest row.
    """

    fp = Path(row.file_path)
    if not fp.is_absolute():
        fp = (repo_root / fp).resolve()
    if not fp.exists():
        raise FileNotFoundError(f"External source file not found: {fp}")

    got = sha256_file_bytes(fp)
    if got != row.sha256:
        raise ValueError(f"SHA256 mismatch for {row.source_id}: expected={row.sha256} got={got}")

    # Provenance: store sha256 as file_hash (source_document is keyed by file_hash).
    source_doc_id = await create_source_document(
        session,
        file_name=str(fp.name),
        file_hash=row.sha256,
        framework_version="external_corpus_v1",
    )
    run_id = await create_ingestion_run(session, source_doc_id)

    text = _read_text(fp)
    paras = _split_paragraphs(text)
    chunk_texts = _chunk_text(paras)

    chunks_inserted = 0
    for i, ct in enumerate(chunk_texts):
        chunk_id = _stable_chunk_id(source_id=row.source_id, idx=i, text=ct)
        # Store as evidence chunks. Entity is the external source itself.
        await session.execute(
            # Use text(...) lazily to keep this module import-light.
            __import__("sqlalchemy").text(
                """
                INSERT INTO chunk (
                    chunk_id, entity_type, entity_id, chunk_type,
                    text_ar, text_en, source_doc_id, source_anchor, token_count_estimate
                )
                VALUES (
                    :chunk_id, 'external_source', :entity_id, 'evidence',
                    :text_ar, NULL, :source_doc_id, :source_anchor, :tok
                )
                ON CONFLICT (chunk_id) DO UPDATE SET
                    text_ar = EXCLUDED.text_ar,
                    source_anchor = EXCLUDED.source_anchor
                """
            ),
            {
                "chunk_id": chunk_id,
                "entity_id": row.source_id,
                "text_ar": ct,
                "source_doc_id": source_doc_id,
                "source_anchor": f"external:{row.source_id}:p{i:03d}",
                "tok": int(max(1, len(ct.split()))),
            },
        )
        chunks_inserted += 1

    embeddings = 0
    if embed:
        try:
            embeddings = await embed_all_chunks_for_source(session=session, source_doc_id=source_doc_id)
        except Exception:
            embeddings = 0

    await complete_ingestion_run(
        session,
        run_id,
        entities_extracted=0,
        evidence_extracted=chunks_inserted,
        validation_errors=[],
        status="completed",
    )

    return ExternalIngestResult(
        source_id=row.source_id,
        source_doc_id=source_doc_id,
        run_id=run_id,
        chunks=chunks_inserted,
        embeddings=embeddings,
    )


async def ingest_external_sources(
    *,
    session: AsyncSession,
    rows: list[ExternalSourceManifestRow],
    repo_root: Path,
    embed: bool = True,
) -> list[ExternalIngestResult]:
    out: list[ExternalIngestResult] = []
    for r in rows:
        out.append(await ingest_external_source(session=session, row=r, repo_root=repo_root, embed=embed))
    return out

