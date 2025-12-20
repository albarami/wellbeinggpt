"""Ingestion loader: metadata + run tracking.

Reason: keep each module <500 LOC.
"""

from __future__ import annotations

import inspect
import json
import uuid
from datetime import datetime
from typing import Any, Optional

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession


async def _scalar_one_or_none(session: AsyncSession, sql: str, params: dict[str, Any]) -> Optional[str]:
    """Best-effort scalar fetch helper."""
    result = await session.execute(text(sql), params)
    try:
        val = result.scalar_one_or_none()
        if inspect.isawaitable(val):  # AsyncMock in unit tests
            val = await val
    except Exception:
        return None
    if isinstance(val, uuid.UUID):
        return str(val)
    if isinstance(val, str):
        return val
    return None


async def create_source_document(
    session: AsyncSession,
    file_name: str,
    file_hash: str,
    framework_version: str,
) -> str:
    """Create (or upsert) a source document record and return its id."""
    doc_id = str(uuid.uuid4())
    result = await session.execute(
        text(
            """
            INSERT INTO source_document (id, file_name, file_hash, framework_version)
            VALUES (:id, :file_name, :file_hash, :framework_version)
            ON CONFLICT (file_hash) DO UPDATE SET
                file_name = EXCLUDED.file_name,
                framework_version = EXCLUDED.framework_version
            RETURNING id
            """
        ),
        {"id": doc_id, "file_name": file_name, "file_hash": file_hash, "framework_version": framework_version},
    )
    persisted_id = result.scalar_one()
    if inspect.isawaitable(persisted_id):  # AsyncMock in unit tests
        persisted_id = await persisted_id
    return str(persisted_id)


async def create_ingestion_run(session: AsyncSession, source_doc_id: str) -> str:
    """Create an ingestion run record."""
    run_id = str(uuid.uuid4())
    await session.execute(
        text("INSERT INTO ingestion_run (id, source_doc_id, status) VALUES (:id, :source_doc_id, 'in_progress')"),
        {"id": run_id, "source_doc_id": source_doc_id},
    )
    return run_id


async def complete_ingestion_run(
    session: AsyncSession,
    run_id: str,
    entities_extracted: int,
    evidence_extracted: int,
    validation_errors: list[str],
    status: str = "completed",
) -> None:
    """Mark an ingestion run as complete."""
    await session.execute(
        text(
            """
            UPDATE ingestion_run
            SET status = :status,
                entities_extracted = :entities,
                evidence_extracted = :evidence,
                validation_errors = :errors,
                completed_at = :completed_at
            WHERE id = :id
            """
        ),
        {
            "id": run_id,
            "status": status,
            "entities": entities_extracted,
            "evidence": evidence_extracted,
            "errors": json.dumps(validation_errors),
            "completed_at": datetime.utcnow(),
        },
    )

