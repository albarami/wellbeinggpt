"""Ingestion loader: chunk + embedding ingestion.

Reason: keep each module <500 LOC.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Optional

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from apps.api.llm.embedding_client_azure import AzureEmbeddingClient, EmbeddingConfig
from apps.api.retrieve.azure_search_indexer import (
    chunk_doc as azure_search_chunk_doc,
    ensure_index as ensure_azure_search_index,
    is_configured as azure_search_is_configured,
    upsert_documents as azure_search_upsert_documents,
)
from apps.api.retrieve.vector_retriever import store_embedding


async def load_chunks_jsonl(
    session: AsyncSession,
    chunks_jsonl_path: str,
    source_doc_id: str,
    run_id: str,
    id_maps: Optional[dict[str, dict[str, str]]] = None,
) -> int:
    """Load chunks JSONL (Evidence Packets) into chunk + chunk_ref tables."""
    path = Path(chunks_jsonl_path)
    if not path.exists():
        return 0

    count = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            entity_type = row.get("entity_type", "")
            entity_id = row.get("entity_id", "")
            if id_maps and entity_type in id_maps and entity_id in id_maps[entity_type]:
                entity_id = id_maps[entity_type][entity_id]
            await session.execute(
                text(
                    """
                    INSERT INTO chunk (chunk_id, entity_type, entity_id, chunk_type, text_ar, text_en,
                                       source_doc_id, source_anchor, token_count_estimate)
                    VALUES (:chunk_id, :entity_type, :entity_id, :chunk_type, :text_ar, :text_en,
                            :source_doc_id, :source_anchor, :token_count_estimate)
                    ON CONFLICT (chunk_id) DO UPDATE SET
                        text_ar = EXCLUDED.text_ar,
                        source_anchor = EXCLUDED.source_anchor
                    """
                ),
                {
                    "chunk_id": row["chunk_id"],
                    "entity_type": entity_type,
                    "entity_id": entity_id,
                    "chunk_type": row["chunk_type"],
                    "text_ar": row.get("text_ar", ""),
                    "text_en": row.get("text_en"),
                    "source_doc_id": source_doc_id,
                    "source_anchor": row.get("source_anchor", ""),
                    "token_count_estimate": int(row.get("token_count_estimate") or 0),
                },
            )
            for r in row.get("refs", []) or []:
                await session.execute(
                    text(
                        """
                        INSERT INTO chunk_ref (chunk_id, ref_type, ref)
                        VALUES (:chunk_id, :ref_type, :ref)
                        ON CONFLICT DO NOTHING
                        """
                    ),
                    {"chunk_id": row["chunk_id"], "ref_type": r.get("type", ""), "ref": r.get("ref", "")},
                )
            count += 1
    return count


async def embed_all_chunks_for_source(
    session: AsyncSession,
    source_doc_id: str,
    batch_size: int = 64,
) -> int:
    """Embed all chunks for a given source_doc_id and upsert into embedding table."""
    cfg = EmbeddingConfig.from_env()
    if not cfg.is_configured():
        return 0
    client = AzureEmbeddingClient(cfg)

    vector_backend = os.getenv("VECTOR_BACKEND", "disabled").lower()
    azure_search_enabled = vector_backend == "azure_search" and azure_search_is_configured()
    if azure_search_enabled:
        await ensure_azure_search_index(cfg.dims)

    result = await session.execute(
        text(
            """
            SELECT chunk_id, entity_type, entity_id, chunk_type, text_ar, source_anchor
            FROM chunk
            WHERE source_doc_id = :source_doc_id
            ORDER BY chunk_id
            """
        ),
        {"source_doc_id": source_doc_id},
    )
    rows = result.fetchall()
    total = 0
    for i in range(0, len(rows), batch_size):
        batch = rows[i : i + batch_size]
        texts = [r.text_ar or "" for r in batch]
        vecs = await client.embed_texts(texts)

        chunk_ids = [r.chunk_id for r in batch]
        refs_by_chunk: dict[str, list[dict[str, Any]]] = {str(cid): [] for cid in chunk_ids}
        ref_rows = (
            await session.execute(
                text(
                    """
                    SELECT chunk_id, ref_type, ref
                    FROM chunk_ref
                    WHERE chunk_id = ANY(:chunk_ids)
                    """
                ),
                {"chunk_ids": chunk_ids},
            )
        ).fetchall()
        for rr in ref_rows:
            refs_by_chunk[str(rr.chunk_id)].append({"type": rr.ref_type, "ref": rr.ref})

        azure_docs: list[dict[str, Any]] = []
        for r, v in zip(batch, vecs):
            await store_embedding(session=session, chunk_id=r.chunk_id, vector=v, model=cfg.embedding_deployment, dims=cfg.dims)
            if azure_search_enabled:
                azure_docs.append(
                    azure_search_chunk_doc(
                        chunk_id=str(r.chunk_id),
                        entity_type=str(r.entity_type),
                        entity_id=str(r.entity_id),
                        chunk_type=str(r.chunk_type),
                        text_ar=str(r.text_ar or ""),
                        source_doc_id=str(source_doc_id),
                        source_anchor=str(r.source_anchor or ""),
                        refs=refs_by_chunk.get(str(r.chunk_id), []),
                        vector=[float(x) for x in v],
                    )
                )
            total += 1

        if azure_search_enabled and azure_docs:
            await azure_search_upsert_documents(azure_docs)
    return total

