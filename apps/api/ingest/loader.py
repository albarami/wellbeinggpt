"""
Database Loader Module

Loads canonical JSON into PostgreSQL database with versioning and provenance.
"""

import json
import inspect
import uuid
from datetime import datetime
from typing import Any, Optional

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from apps.api.llm.embedding_client_azure import AzureEmbeddingClient, EmbeddingConfig
from apps.api.retrieve.vector_retriever import store_embedding
from apps.api.retrieve.azure_search_indexer import (
    ensure_index as ensure_azure_search_index,
    upsert_documents as azure_search_upsert_documents,
    chunk_doc as azure_search_chunk_doc,
    is_configured as azure_search_is_configured,
)
import os


async def _scalar_one_or_none(
    session: AsyncSession, sql: str, params: dict[str, Any]
) -> Optional[str]:
    """
    Best-effort scalar fetch helper.

    Notes:
    - Returns None on missing rows or when running under mock sessions in unit tests.
    """
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
    """
    Create a source document record.

    Args:
        session: Database session.
        file_name: Original file name.
        file_hash: SHA256 hash of file.
        framework_version: Framework version string.

    Returns:
        The source document ID.
    """
    doc_id = str(uuid.uuid4())

    result = await session.execute(
        text("""
            INSERT INTO source_document (id, file_name, file_hash, framework_version)
            VALUES (:id, :file_name, :file_hash, :framework_version)
            ON CONFLICT (file_hash) DO UPDATE SET
                file_name = EXCLUDED.file_name,
                framework_version = EXCLUDED.framework_version
            RETURNING id
        """),
        {
            "id": doc_id,
            "file_name": file_name,
            "file_hash": file_hash,
            "framework_version": framework_version,
        }
    )
    # Important: when ON CONFLICT triggers, the inserted id is ignored; RETURNING yields the existing id.
    persisted_id = result.scalar_one()
    if inspect.isawaitable(persisted_id):  # AsyncMock in unit tests
        persisted_id = await persisted_id
    return str(persisted_id)


async def create_ingestion_run(
    session: AsyncSession,
    source_doc_id: str,
) -> str:
    """
    Create an ingestion run record.

    Args:
        session: Database session.
        source_doc_id: Source document ID.

    Returns:
        The ingestion run ID.
    """
    run_id = str(uuid.uuid4())

    await session.execute(
        text("""
            INSERT INTO ingestion_run (id, source_doc_id, status)
            VALUES (:id, :source_doc_id, 'in_progress')
        """),
        {"id": run_id, "source_doc_id": source_doc_id}
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
    """
    Mark an ingestion run as complete.

    Args:
        session: Database session.
        run_id: Ingestion run ID.
        entities_extracted: Count of entities.
        evidence_extracted: Count of evidence.
        validation_errors: List of validation error messages.
        status: Final status.
    """
    await session.execute(
        text("""
            UPDATE ingestion_run
            SET status = :status,
                entities_extracted = :entities,
                evidence_extracted = :evidence,
                validation_errors = :errors,
                completed_at = :completed_at
            WHERE id = :id
        """),
        {
            "id": run_id,
            "status": status,
            "entities": entities_extracted,
            "evidence": evidence_extracted,
            "errors": json.dumps(validation_errors),
            "completed_at": datetime.utcnow(),
        }
    )


async def load_pillar(
    session: AsyncSession,
    pillar_data: dict[str, Any],
    source_doc_id: str,
    run_id: str,
) -> None:
    """
    Load a pillar into the database.

    Args:
        session: Database session.
        pillar_data: Pillar dictionary from canonical JSON.
        source_doc_id: Source document ID.
        run_id: Ingestion run ID.
    """
    # Idempotency:
    # Prefer existing row by natural key (source_doc_id, name_ar) if present.
    resolved_id = await _scalar_one_or_none(
        session,
        "SELECT id FROM pillar WHERE source_doc_id = :sd AND name_ar = :n",
        {"sd": source_doc_id, "n": pillar_data["name_ar"]},
    ) or pillar_data["id"]

    await session.execute(
        text("""
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
        """),
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
        }
    )

    # Also store an anchored text_block for any supplemental OCR blocks attached to this pillar.
    # Reason: Pillar.description_ar is not anchor-scoped; screenshots must remain auditable via userimg_* anchors.
    # (These blocks are loaded in load_canonical_json_to_db from canonical["supplemental_text_blocks"].)
    return resolved_id


async def load_core_value(
    session: AsyncSession,
    cv_data: dict[str, Any],
    pillar_id: str,
    source_doc_id: str,
    run_id: str,
) -> None:
    """
    Load a core value into the database.

    Args:
        session: Database session.
        cv_data: Core value dictionary.
        pillar_id: Parent pillar ID.
        source_doc_id: Source document ID.
        run_id: Ingestion run ID.
    """
    # Get definition text if present
    definition_ar = None
    if cv_data.get("definition"):
        definition_ar = cv_data["definition"].get("text_ar")

    resolved_id = await _scalar_one_or_none(
        session,
        "SELECT id FROM core_value WHERE pillar_id = :pid AND name_ar = :n",
        {"pid": pillar_id, "n": cv_data["name_ar"]},
    ) or cv_data["id"]

    await session.execute(
        text("""
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
        """),
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
        }
    )

    # Load text block for definition
    if definition_ar:
        await load_text_block(
            session,
            entity_type="core_value",
            entity_id=resolved_id,
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
    return resolved_id


async def load_sub_value(
    session: AsyncSession,
    sv_data: dict[str, Any],
    core_value_id: str,
    source_doc_id: str,
    run_id: str,
) -> None:
    """
    Load a sub-value into the database.

    Args:
        session: Database session.
        sv_data: Sub-value dictionary.
        core_value_id: Parent core value ID.
        source_doc_id: Source document ID.
        run_id: Ingestion run ID.
    """
    definition_ar = None
    if sv_data.get("definition"):
        definition_ar = sv_data["definition"].get("text_ar")

    resolved_id = await _scalar_one_or_none(
        session,
        "SELECT id FROM sub_value WHERE core_value_id = :cid AND name_ar = :n",
        {"cid": core_value_id, "n": sv_data["name_ar"]},
    ) or sv_data["id"]

    await session.execute(
        text("""
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
        """),
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
        }
    )

    # Load text block for definition
    if definition_ar:
        await load_text_block(
            session,
            entity_type="sub_value",
            entity_id=resolved_id,
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
    return resolved_id


async def load_text_block(
    session: AsyncSession,
    entity_type: str,
    entity_id: str,
    block_type: str,
    text_ar: str,
    source_doc_id: str,
    source_anchor: dict,
    run_id: str,
) -> str:
    """
    Load a text block into the database.

    Args:
        session: Database session.
        entity_type: Type of parent entity.
        entity_id: ID of parent entity.
        block_type: Type of block (definition, commentary).
        text_ar: Arabic text content.
        source_doc_id: Source document ID.
        source_anchor: Anchor information.
        run_id: Ingestion run ID.

    Returns:
        The text block ID.
    """
    # Idempotency:
    # - For normal blocks, upsert by (source_doc_id, entity_type, entity_id, block_type).
    # - For supplemental OCR audit blocks, allow MULTIPLE blocks per entity (one per screenshot anchor),
    #   and dedupe by (source_doc_id, entity_type, entity_id, block_type, source_anchor->>source_anchor).
    params = {"sd": source_doc_id, "et": entity_type, "eid": entity_id, "bt": block_type}
    if block_type == "supplemental_ocr":
        anchor_str = ""
        if isinstance(source_anchor, dict):
            anchor_str = str(source_anchor.get("source_anchor") or "")
        params["a"] = anchor_str
        existing_id = await _scalar_one_or_none(
            session,
            """
            SELECT id
            FROM text_block
            WHERE source_doc_id = :sd
              AND entity_type = :et
              AND entity_id = :eid
              AND block_type = :bt
              AND (source_anchor->>'source_anchor') = :a
            """,
            params,
        )
    else:
        existing_id = await _scalar_one_or_none(
            session,
            """
            SELECT id
            FROM text_block
            WHERE source_doc_id = :sd
              AND entity_type = :et
              AND entity_id = :eid
              AND block_type = :bt
            """,
            params,
        )
    if existing_id:
        await session.execute(
            text(
                """
                UPDATE text_block
                SET text_ar = :text_ar,
                    source_anchor = :source_anchor,
                    ingestion_run_id = :run_id
                WHERE id = :id
                """
            ),
            {
                "id": existing_id,
                "text_ar": text_ar,
                "source_anchor": json.dumps(source_anchor),
                "run_id": run_id,
            },
        )
        return existing_id

    block_id = str(uuid.uuid4())
    await session.execute(
        text(
            """
            INSERT INTO text_block (id, entity_type, entity_id, block_type, text_ar,
                                   source_doc_id, source_anchor, ingestion_run_id)
            VALUES (:id, :entity_type, :entity_id, :block_type, :text_ar,
                    :source_doc_id, :source_anchor, :run_id)
            """
        ),
        {
            "id": block_id,
            "entity_type": entity_type,
            "entity_id": entity_id,
            "block_type": block_type,
            "text_ar": text_ar,
            "source_doc_id": source_doc_id,
            "source_anchor": json.dumps(source_anchor),
            "run_id": run_id,
        },
    )
    return block_id


async def load_evidence(
    session: AsyncSession,
    entity_type: str,
    entity_id: str,
    evidence_type: str,
    ref_raw: str,
    ref_norm: Optional[str],
    text_ar: str,
    source_doc_id: str,
    source_anchor: dict,
    run_id: str,
    parse_status: str = "success",
    surah_name_ar: Optional[str] = None,
    surah_number: Optional[int] = None,
    ayah_number: Optional[int] = None,
    hadith_collection: Optional[str] = None,
    hadith_number: Optional[int] = None,
) -> str:
    """
    Load an evidence record into the database.

    Args:
        session: Database session.
        entity_type: Type of parent entity.
        entity_id: ID of parent entity.
        evidence_type: Type of evidence (quran, hadith, book).
        ref_raw: Original reference string.
        ref_norm: Normalized reference.
        text_ar: Evidence text.
        source_doc_id: Source document ID.
        source_anchor: Anchor information.
        run_id: Ingestion run ID.
        parse_status: Parse status.
        surah_name_ar: Surah name for Quran refs.
        surah_number: Surah number for Quran refs.
        ayah_number: Ayah number for Quran refs.
        hadith_collection: Collection name for Hadith refs.
        hadith_number: Hadith number.

    Returns:
        The evidence ID.
    """
    # Idempotent dedupe by (source_doc_id, entity, evidence_type, ref_norm/ref_raw, text_ar hash).
    existing_id = await _scalar_one_or_none(
        session,
        """
        SELECT id
        FROM evidence
        WHERE source_doc_id = :sd
          AND entity_type = :et
          AND entity_id = :eid
          AND evidence_type = :ev
          AND COALESCE(ref_norm,'') = COALESCE(:ref_norm,'')
          AND md5(text_ar) = md5(:text_ar)
        """,
        {
            "sd": source_doc_id,
            "et": entity_type,
            "eid": entity_id,
            "ev": evidence_type,
            "ref_norm": ref_norm or "",
            "text_ar": text_ar,
        },
    )
    if existing_id:
        return existing_id

    # Defensive normalization for enterprise DB constraints:
    # - OCR may produce long "ref_norm" or misparsed "surah_name_ar"/collection strings.
    # - For non-success parses, we store only raw fields and clear structured columns.
    if (parse_status or "").lower() != "success":
        ref_norm = None
        surah_name_ar = None
        surah_number = None
        ayah_number = None
        hadith_collection = None
        hadith_number = None
    else:
        if ref_norm and len(ref_norm) > 255:
            # Avoid truncation + false SHARES_REF edges.
            ref_norm = None
            parse_status = "needs_review"
            surah_name_ar = None
            surah_number = None
            ayah_number = None
            hadith_collection = None
            hadith_number = None
        if surah_name_ar and len(surah_name_ar) > 100:
            parse_status = "needs_review"
            surah_name_ar = None
            surah_number = None
            ayah_number = None
        if hadith_collection and len(hadith_collection) > 100:
            parse_status = "needs_review"
            hadith_collection = None
            hadith_number = None

    evidence_id = str(uuid.uuid4())

    await session.execute(
        text("""
            INSERT INTO evidence (id, entity_type, entity_id, evidence_type,
                                 ref_raw, ref_norm, text_ar,
                                 surah_name_ar, surah_number, ayah_number,
                                 hadith_collection, hadith_number,
                                 parse_status, source_doc_id, source_anchor,
                                 ingestion_run_id)
            VALUES (:id, :entity_type, :entity_id, :evidence_type,
                    :ref_raw, :ref_norm, :text_ar,
                    :surah_name_ar, :surah_number, :ayah_number,
                    :hadith_collection, :hadith_number,
                    :parse_status, :source_doc_id, :source_anchor,
                    :run_id)
        """),
        {
            "id": evidence_id,
            "entity_type": entity_type,
            "entity_id": entity_id,
            "evidence_type": evidence_type,
            "ref_raw": ref_raw,
            "ref_norm": ref_norm,
            "text_ar": text_ar,
            "surah_name_ar": surah_name_ar,
            "surah_number": surah_number,
            "ayah_number": ayah_number,
            "hadith_collection": hadith_collection,
            "hadith_number": hadith_number,
            "parse_status": parse_status,
            "source_doc_id": source_doc_id,
            "source_anchor": json.dumps(source_anchor),
            "run_id": run_id,
        }
    )

    return evidence_id


async def load_canonical_json_to_db(
    session: AsyncSession,
    canonical_data: dict[str, Any],
    file_name: str,
) -> dict[str, Any]:
    """
    Load canonical JSON data into the database.

    This is the main entry point for loading extracted data.

    Args:
        session: Database session.
        canonical_data: Canonical JSON data.
        file_name: Original file name.

    Returns:
        Summary of loaded data.
    """
    meta = canonical_data.get("meta", {})
    file_hash = meta.get("source_file_hash", "unknown")
    framework_version = meta.get("framework_version", "unknown")

    # Create source document
    source_doc_id = await create_source_document(
        session, file_name, file_hash, framework_version
    )

    # Create ingestion run
    run_id = await create_ingestion_run(session, source_doc_id)

    # Enterprise-grade default: purge existing derived rows for this source_doc_id before re-ingestion.
    # Reason: DOCX is the source of truth; ingestion must be repeatable and converge to the same DB state.
    purge = os.getenv("INGEST_PURGE_EXISTING", "true").lower() in ("1", "true", "yes")
    if purge:
        # Delete in dependency-safe order.
        for stmt in [
            # Edges can point to any entity type; remove all edges involving entities from this source_doc_id.
            "DELETE FROM edge WHERE from_id IN (SELECT id FROM pillar WHERE source_doc_id=:sd) OR to_id IN (SELECT id FROM pillar WHERE source_doc_id=:sd)",
            "DELETE FROM edge WHERE from_id IN (SELECT id FROM core_value WHERE source_doc_id=:sd) OR to_id IN (SELECT id FROM core_value WHERE source_doc_id=:sd)",
            "DELETE FROM edge WHERE from_id IN (SELECT id FROM sub_value WHERE source_doc_id=:sd) OR to_id IN (SELECT id FROM sub_value WHERE source_doc_id=:sd)",
            # evidence.id is UUID while edge.*_id is VARCHAR; cast UUIDs to text for comparison.
            "DELETE FROM edge WHERE from_id IN (SELECT id::text FROM evidence WHERE source_doc_id=:sd) OR to_id IN (SELECT id::text FROM evidence WHERE source_doc_id=:sd)",
            "DELETE FROM edge WHERE from_id IN (SELECT chunk_id FROM chunk WHERE source_doc_id=:sd) OR to_id IN (SELECT chunk_id FROM chunk WHERE source_doc_id=:sd)",
            "DELETE FROM chunk_ref WHERE chunk_id IN (SELECT chunk_id FROM chunk WHERE source_doc_id=:sd)",
            "DELETE FROM embedding WHERE chunk_id IN (SELECT chunk_id FROM chunk WHERE source_doc_id=:sd)",
            "DELETE FROM chunk WHERE source_doc_id=:sd",
            "DELETE FROM evidence WHERE source_doc_id=:sd",
            "DELETE FROM text_block WHERE source_doc_id=:sd",
            "DELETE FROM sub_value WHERE source_doc_id=:sd",
            "DELETE FROM core_value WHERE source_doc_id=:sd",
            "DELETE FROM pillar WHERE source_doc_id=:sd",
        ]:
            try:
                await session.execute(text(stmt), {"sd": source_doc_id})
            except Exception:
                continue

    # Load pillars and their contents
    pillars = canonical_data.get("pillars", [])
    total_cv = 0
    total_sv = 0
    total_evidence = 0

    # ID maps: canonical id -> persisted id
    pillar_id_map: dict[str, str] = {}
    core_id_map: dict[str, str] = {}
    sub_id_map: dict[str, str] = {}

    for pillar_data in pillars:
        pillar_db_id = await load_pillar(session, pillar_data, source_doc_id, run_id)
        pillar_id_map[str(pillar_data["id"])] = str(pillar_db_id)

        for cv_data in pillar_data.get("core_values", []):
            cv_db_id = await load_core_value(
                session, cv_data, pillar_db_id, source_doc_id, run_id
            )
            core_id_map[str(cv_data["id"])] = str(cv_db_id)
            total_cv += 1

            # Load evidence for core value
            for ev in cv_data.get("evidence", []) or []:
                await load_evidence(
                    session=session,
                    entity_type="core_value",
                    entity_id=cv_db_id,
                    evidence_type=ev.get("evidence_type", "book"),
                    ref_raw=ev.get("ref_raw", "") or "",
                    ref_norm=ev.get("ref_norm"),
                    text_ar=ev.get("text_ar", "") or "",
                    source_doc_id=source_doc_id,
                    source_anchor={"source_anchor": ev.get("source_anchor", "")},
                    run_id=run_id,
                    parse_status=ev.get("parse_status", "success"),
                    surah_name_ar=ev.get("surah_name_ar"),
                    surah_number=ev.get("surah_number"),
                    ayah_number=ev.get("ayah_number"),
                    hadith_collection=ev.get("hadith_collection"),
                    hadith_number=ev.get("hadith_number"),
                )
                total_evidence += 1

            for sv_data in cv_data.get("sub_values", []):
                sv_db_id = await load_sub_value(
                    session, sv_data, cv_db_id, source_doc_id, run_id
                )
                sub_id_map[str(sv_data["id"])] = str(sv_db_id)
                total_sv += 1

                # Load evidence for sub value
                for ev in sv_data.get("evidence", []) or []:
                    await load_evidence(
                        session=session,
                        entity_type="sub_value",
                        entity_id=sv_db_id,
                        evidence_type=ev.get("evidence_type", "book"),
                        ref_raw=ev.get("ref_raw", "") or "",
                        ref_norm=ev.get("ref_norm"),
                        text_ar=ev.get("text_ar", "") or "",
                        source_doc_id=source_doc_id,
                        source_anchor={"source_anchor": ev.get("source_anchor", "")},
                        run_id=run_id,
                        parse_status=ev.get("parse_status", "success"),
                        surah_name_ar=ev.get("surah_name_ar"),
                        surah_number=ev.get("surah_number"),
                        ayah_number=ev.get("ayah_number"),
                        hadith_collection=ev.get("hadith_collection"),
                        hadith_number=ev.get("hadith_number"),
                    )
                    total_evidence += 1

    # Load anchored "supplemental OCR" blocks (audit trail for user-provided screenshots).
    # These blocks are intentionally stored separately from extracted definitions/evidence.
    for b in canonical_data.get("supplemental_text_blocks", []) or []:
        try:
            et = str(b.get("entity_type") or "")
            eid = str(b.get("entity_id") or "")
            bt = str(b.get("block_type") or "supplemental_ocr")
            text_ar = str(b.get("text_ar") or "").strip()
            source_anchor = b.get("source_anchor") or {}
            if not et or not eid or not text_ar:
                continue

            # Map canonical IDs to persisted IDs when maps exist.
            if et == "pillar" and eid in pillar_id_map:
                eid = pillar_id_map[eid]
            elif et == "core_value" and eid in core_id_map:
                eid = core_id_map[eid]
            elif et == "sub_value" and eid in sub_id_map:
                eid = sub_id_map[eid]

            await load_text_block(
                session=session,
                entity_type=et,
                entity_id=eid,
                block_type=bt,
                text_ar=text_ar,
                source_doc_id=source_doc_id,
                source_anchor=source_anchor if isinstance(source_anchor, dict) else {"source_anchor": str(source_anchor)},
                run_id=run_id,
            )
        except Exception:
            continue

    # Load chunks from generated JSONL if provided in meta (optional)
    chunks_path = canonical_data.get("meta", {}).get("chunks_path")
    total_chunks = 0
    total_embeddings = 0
    if chunks_path:
        total_chunks = await load_chunks_jsonl(
            session=session,
            chunks_jsonl_path=chunks_path,
            source_doc_id=source_doc_id,
            run_id=run_id,
            id_maps={"pillar": pillar_id_map, "core_value": core_id_map, "sub_value": sub_id_map},
        )
        # Embed chunks (best-effort; if not configured, skip)
        try:
            total_embeddings = await embed_all_chunks_for_source(
                session=session,
                source_doc_id=source_doc_id,
            )
        except Exception:
            total_embeddings = 0

    # Build graph edges (hierarchy + evidence links)
    await build_edges_for_source(session=session, source_doc_id=source_doc_id)

    # Complete ingestion run
    await complete_ingestion_run(
        session,
        run_id,
        entities_extracted=len(pillars) + total_cv + total_sv,
        evidence_extracted=total_evidence,
        validation_errors=meta.get("validation_errors", []),
    )

    return {
        "source_doc_id": source_doc_id,
        "run_id": run_id,
        "pillars": len(pillars),
        "core_values": total_cv,
        "sub_values": total_sv,
        "evidence": total_evidence,
        "chunks": total_chunks,
        "embeddings": total_embeddings,
    }


async def load_chunks_jsonl(
    session: AsyncSession,
    chunks_jsonl_path: str,
    source_doc_id: str,
    run_id: str,
    id_maps: Optional[dict[str, dict[str, str]]] = None,
) -> int:
    """
    Load chunks JSONL (Evidence Packets) into chunk + chunk_ref tables.
    """
    from pathlib import Path

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
            # chunk refs
            for r in row.get("refs", []) or []:
                await session.execute(
                    text(
                        """
                        INSERT INTO chunk_ref (chunk_id, ref_type, ref)
                        VALUES (:chunk_id, :ref_type, :ref)
                        ON CONFLICT DO NOTHING
                        """
                    ),
                    {
                        "chunk_id": row["chunk_id"],
                        "ref_type": r.get("type", ""),
                        "ref": r.get("ref", ""),
                    },
                )
            count += 1
    return count


async def embed_all_chunks_for_source(
    session: AsyncSession,
    source_doc_id: str,
    batch_size: int = 64,
) -> int:
    """
    Embed all chunks for a given source_doc_id and upsert into embedding table.
    """
    cfg = EmbeddingConfig.from_env()
    if not cfg.is_configured():
        return 0
    client = AzureEmbeddingClient(cfg)

    vector_backend = os.getenv("VECTOR_BACKEND", "disabled").lower()
    azure_search_enabled = vector_backend == "azure_search" and azure_search_is_configured()
    if azure_search_enabled:
        # Best-effort ensure index exists
        await ensure_azure_search_index(cfg.dims)

    # Fetch chunks
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

        # Pre-fetch refs for this batch (chunk_ref table)
        chunk_ids = [r.chunk_id for r in batch]
        refs_by_chunk: dict[str, list[dict[str, Any]]] = {cid: [] for cid in chunk_ids}
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
            await store_embedding(
                session=session,
                chunk_id=r.chunk_id,
                vector=v,
                model=cfg.embedding_deployment,
                dims=cfg.dims,
            )
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


async def build_edges_for_source(session: AsyncSession, source_doc_id: str) -> None:
    """
    Build minimal graph edges in Postgres:
    - Pillar CONTAINS CoreValue
    - CoreValue CONTAINS SubValue
    - (Core/Sub) SUPPORTED_BY Evidence
    - (Core/Sub) SHARES_REF (cross-links) when multiple entities share the same ref_norm
    """
    # CONTAINS: pillar -> core_value
    await session.execute(
        text(
            """
            INSERT INTO edge (from_type, from_id, rel_type, to_type, to_id,
                              created_method, created_by, justification, status)
            SELECT 'pillar', cv.pillar_id, 'CONTAINS', 'core_value', cv.id,
                   'rule_exact_match', 'system', 'hierarchy', 'approved'
            FROM core_value cv
            WHERE cv.source_doc_id = :source_doc_id
            ON CONFLICT DO NOTHING
            """
        ),
        {"source_doc_id": source_doc_id},
    )

    # CONTAINS: core_value -> sub_value
    await session.execute(
        text(
            """
            INSERT INTO edge (from_type, from_id, rel_type, to_type, to_id,
                              created_method, created_by, justification, status)
            SELECT 'core_value', sv.core_value_id, 'CONTAINS', 'sub_value', sv.id,
                   'rule_exact_match', 'system', 'hierarchy', 'approved'
            FROM sub_value sv
            WHERE sv.source_doc_id = :source_doc_id
            ON CONFLICT DO NOTHING
            """
        ),
        {"source_doc_id": source_doc_id},
    )

    # SUPPORTED_BY: entity -> evidence
    await session.execute(
        text(
            """
            INSERT INTO edge (from_type, from_id, rel_type, to_type, to_id,
                              created_method, created_by, justification, status)
            SELECT e.entity_type, e.entity_id, 'SUPPORTED_BY', 'evidence', e.id::text,
                   'rule_exact_match', 'system', 'evidence_link', 'approved'
            FROM evidence e
            WHERE e.source_doc_id = :source_doc_id
            ON CONFLICT DO NOTHING
            """
        ),
        {"source_doc_id": source_doc_id},
    )

    # SHARES_REF: cross-link entities that share the same normalized reference.
    # This creates an explicit graph signal for "same verse/hadith used in multiple places",
    # enabling cross-pillar discovery without relying on vector-only coincidence.
    await session.execute(
        text(
            """
            WITH refs AS (
                SELECT DISTINCT
                    e.entity_type,
                    e.entity_id,
                    e.ref_norm,
                    e.evidence_type
                FROM evidence e
                WHERE e.source_doc_id = :source_doc_id
                  AND e.ref_norm IS NOT NULL
                  AND e.ref_norm <> ''
            ),
            pairs AS (
                SELECT
                    r1.entity_type AS from_type,
                    r1.entity_id AS from_id,
                    r2.entity_type AS to_type,
                    r2.entity_id AS to_id,
                    r1.ref_norm AS ref_norm,
                    r1.evidence_type AS evidence_type
                FROM refs r1
                JOIN refs r2
                  ON r1.ref_norm = r2.ref_norm
                 AND (r1.entity_type, r1.entity_id) < (r2.entity_type, r2.entity_id)
            )
            INSERT INTO edge (from_type, from_id, rel_type, to_type, to_id,
                              created_method, created_by, justification, status)
            SELECT
                p.from_type,
                p.from_id,
                'SHARES_REF',
                p.to_type,
                p.to_id,
                'rule_exact_match',
                'system',
                p.evidence_type || ':' || p.ref_norm,
                'approved'
            FROM pairs p
            ON CONFLICT DO NOTHING
            """
        ),
        {"source_doc_id": source_doc_id},
    )

