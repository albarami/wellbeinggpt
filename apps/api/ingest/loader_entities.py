"""Ingestion loader: pillars / core values / sub values.

Reason: keep each module <500 LOC.
"""

from __future__ import annotations

import json
from typing import Any

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from apps.api.ingest.loader_meta import _scalar_one_or_none
from apps.api.ingest.loader_text import load_text_block


async def load_pillar(
    session: AsyncSession,
    pillar_data: dict[str, Any],
    source_doc_id: str,
    run_id: str,
) -> str:
    """Load a pillar into the database and return its persisted id."""
    resolved_id = await _scalar_one_or_none(
        session,
        "SELECT id FROM pillar WHERE source_doc_id = :sd AND name_ar = :n",
        {"sd": source_doc_id, "n": pillar_data["name_ar"]},
    ) or pillar_data["id"]

    await session.execute(
        text(
            """
            INSERT INTO pillar (id, name_ar, name_en, description_ar,
                               source_doc_id, source_anchor, ingestion_run_id)
            VALUES (:id, :name_ar, :name_en, :description_ar,
                    :source_doc_id, :source_anchor, :run_id)
            ON CONFLICT (id) DO UPDATE SET
                name_ar = EXCLUDED.name_ar,
                name_en = EXCLUDED.name_en,
                description_ar = EXCLUDED.description_ar,
                source_doc_id = EXCLUDED.source_doc_id,
                source_anchor = EXCLUDED.source_anchor,
                ingestion_run_id = EXCLUDED.ingestion_run_id,
                updated_at = NOW()
            """
        ),
        {
            "id": resolved_id,
            "name_ar": pillar_data["name_ar"],
            "name_en": pillar_data.get("name_en"),
            "description_ar": pillar_data.get("description_ar"),
            "source_doc_id": source_doc_id,
            "source_anchor": json.dumps(
                {"source_anchor": pillar_data.get("source_anchor")}
                if isinstance(pillar_data.get("source_anchor"), str)
                else (pillar_data.get("source_anchor", {}) or {})
            ),
            "run_id": run_id,
        },
    )
    return str(resolved_id)


async def load_core_value(
    session: AsyncSession,
    cv_data: dict[str, Any],
    pillar_id: str,
    source_doc_id: str,
    run_id: str,
) -> str:
    """Load a core value into the database and return its persisted id."""
    definition_ar = None
    if cv_data.get("definition"):
        definition_ar = cv_data["definition"].get("text_ar")

    resolved_id = await _scalar_one_or_none(
        session,
        "SELECT id FROM core_value WHERE pillar_id = :pid AND name_ar = :n",
        {"pid": pillar_id, "n": cv_data["name_ar"]},
    ) or cv_data["id"]

    await session.execute(
        text(
            """
            INSERT INTO core_value (id, pillar_id, name_ar, name_en, definition_ar,
                                   source_doc_id, source_anchor, ingestion_run_id)
            VALUES (:id, :pillar_id, :name_ar, :name_en, :definition_ar,
                    :source_doc_id, :source_anchor, :run_id)
            ON CONFLICT (id) DO UPDATE SET
                pillar_id = EXCLUDED.pillar_id,
                name_ar = EXCLUDED.name_ar,
                name_en = EXCLUDED.name_en,
                definition_ar = EXCLUDED.definition_ar,
                source_doc_id = EXCLUDED.source_doc_id,
                source_anchor = EXCLUDED.source_anchor,
                ingestion_run_id = EXCLUDED.ingestion_run_id,
                updated_at = NOW()
            """
        ),
        {
            "id": resolved_id,
            "pillar_id": pillar_id,
            "name_ar": cv_data["name_ar"],
            "name_en": cv_data.get("name_en"),
            "definition_ar": definition_ar,
            "source_doc_id": source_doc_id,
            "source_anchor": json.dumps(
                {"source_anchor": cv_data.get("source_anchor")}
                if isinstance(cv_data.get("source_anchor"), str)
                else (cv_data.get("source_anchor", {}) or {})
            ),
            "run_id": run_id,
        },
    )

    if definition_ar:
        await load_text_block(
            session,
            entity_type="core_value",
            entity_id=str(resolved_id),
            block_type="definition",
            text_ar=definition_ar,
            source_doc_id=source_doc_id,
            source_anchor=(
                {"source_anchor": cv_data["definition"].get("source_anchor")}
                if isinstance(cv_data["definition"].get("source_anchor"), str)
                else (cv_data["definition"].get("source_anchor", {}) or {})
            ),
            run_id=run_id,
        )
    return str(resolved_id)


async def load_sub_value(
    session: AsyncSession,
    sv_data: dict[str, Any],
    core_value_id: str,
    source_doc_id: str,
    run_id: str,
) -> str:
    """Load a sub-value into the database and return its persisted id."""
    definition_ar = None
    if sv_data.get("definition"):
        definition_ar = sv_data["definition"].get("text_ar")

    resolved_id = await _scalar_one_or_none(
        session,
        "SELECT id FROM sub_value WHERE core_value_id = :cid AND name_ar = :n",
        {"cid": core_value_id, "n": sv_data["name_ar"]},
    ) or sv_data["id"]

    await session.execute(
        text(
            """
            INSERT INTO sub_value (id, core_value_id, name_ar, name_en, definition_ar,
                                  source_doc_id, source_anchor, ingestion_run_id)
            VALUES (:id, :core_value_id, :name_ar, :name_en, :definition_ar,
                    :source_doc_id, :source_anchor, :run_id)
            ON CONFLICT (id) DO UPDATE SET
                core_value_id = EXCLUDED.core_value_id,
                name_ar = EXCLUDED.name_ar,
                name_en = EXCLUDED.name_en,
                definition_ar = EXCLUDED.definition_ar,
                source_doc_id = EXCLUDED.source_doc_id,
                source_anchor = EXCLUDED.source_anchor,
                ingestion_run_id = EXCLUDED.ingestion_run_id,
                updated_at = NOW()
            """
        ),
        {
            "id": resolved_id,
            "core_value_id": core_value_id,
            "name_ar": sv_data["name_ar"],
            "name_en": sv_data.get("name_en"),
            "definition_ar": definition_ar,
            "source_doc_id": source_doc_id,
            "source_anchor": json.dumps(
                {"source_anchor": sv_data.get("source_anchor")}
                if isinstance(sv_data.get("source_anchor"), str)
                else (sv_data.get("source_anchor", {}) or {})
            ),
            "run_id": run_id,
        },
    )

    if definition_ar:
        await load_text_block(
            session,
            entity_type="sub_value",
            entity_id=str(resolved_id),
            block_type="definition",
            text_ar=definition_ar,
            source_doc_id=source_doc_id,
            source_anchor=(
                {"source_anchor": sv_data["definition"].get("source_anchor")}
                if isinstance(sv_data.get("definition", {}).get("source_anchor"), str)
                else (sv_data.get("definition", {}).get("source_anchor", {}) or {})
            ),
            run_id=run_id,
        )
    return str(resolved_id)

