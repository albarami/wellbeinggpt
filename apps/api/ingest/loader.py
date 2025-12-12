"""
Database Loader Module

Loads canonical JSON into PostgreSQL database with versioning and provenance.
"""

import json
import uuid
from datetime import datetime
from typing import Any, Optional

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession


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

    await session.execute(
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

    return doc_id


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
    await session.execute(
        text("""
            INSERT INTO pillar (id, name_ar, name_en, description_ar,
                               source_doc_id, source_anchor, ingestion_run_id)
            VALUES (:id, :name_ar, :name_en, :description_ar,
                    :source_doc_id, :source_anchor, :run_id)
            ON CONFLICT (id) DO UPDATE SET
                name_ar = EXCLUDED.name_ar,
                description_ar = EXCLUDED.description_ar,
                updated_at = NOW()
        """),
        {
            "id": pillar_data["id"],
            "name_ar": pillar_data["name_ar"],
            "name_en": pillar_data.get("name_en"),
            "description_ar": pillar_data.get("description_ar"),
            "source_doc_id": source_doc_id,
            "source_anchor": json.dumps(pillar_data.get("source_anchor", {})),
            "run_id": run_id,
        }
    )


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

    await session.execute(
        text("""
            INSERT INTO core_value (id, pillar_id, name_ar, name_en, definition_ar,
                                   source_doc_id, source_anchor, ingestion_run_id)
            VALUES (:id, :pillar_id, :name_ar, :name_en, :definition_ar,
                    :source_doc_id, :source_anchor, :run_id)
            ON CONFLICT (id) DO UPDATE SET
                name_ar = EXCLUDED.name_ar,
                definition_ar = EXCLUDED.definition_ar,
                updated_at = NOW()
        """),
        {
            "id": cv_data["id"],
            "pillar_id": pillar_id,
            "name_ar": cv_data["name_ar"],
            "name_en": cv_data.get("name_en"),
            "definition_ar": definition_ar,
            "source_doc_id": source_doc_id,
            "source_anchor": json.dumps(cv_data.get("source_anchor", {})),
            "run_id": run_id,
        }
    )

    # Load text block for definition
    if definition_ar:
        await load_text_block(
            session,
            entity_type="core_value",
            entity_id=cv_data["id"],
            block_type="definition",
            text_ar=definition_ar,
            source_doc_id=source_doc_id,
            source_anchor=cv_data["definition"].get("source_anchor", {}),
            run_id=run_id,
        )


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

    await session.execute(
        text("""
            INSERT INTO sub_value (id, core_value_id, name_ar, name_en, definition_ar,
                                  source_doc_id, source_anchor, ingestion_run_id)
            VALUES (:id, :core_value_id, :name_ar, :name_en, :definition_ar,
                    :source_doc_id, :source_anchor, :run_id)
            ON CONFLICT (id) DO UPDATE SET
                name_ar = EXCLUDED.name_ar,
                definition_ar = EXCLUDED.definition_ar,
                updated_at = NOW()
        """),
        {
            "id": sv_data["id"],
            "core_value_id": core_value_id,
            "name_ar": sv_data["name_ar"],
            "name_en": sv_data.get("name_en"),
            "definition_ar": definition_ar,
            "source_doc_id": source_doc_id,
            "source_anchor": json.dumps(sv_data.get("source_anchor", {})),
            "run_id": run_id,
        }
    )


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
    block_id = str(uuid.uuid4())

    await session.execute(
        text("""
            INSERT INTO text_block (id, entity_type, entity_id, block_type, text_ar,
                                   source_doc_id, source_anchor, ingestion_run_id)
            VALUES (:id, :entity_type, :entity_id, :block_type, :text_ar,
                    :source_doc_id, :source_anchor, :run_id)
        """),
        {
            "id": block_id,
            "entity_type": entity_type,
            "entity_id": entity_id,
            "block_type": block_type,
            "text_ar": text_ar,
            "source_doc_id": source_doc_id,
            "source_anchor": json.dumps(source_anchor),
            "run_id": run_id,
        }
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

    # Load pillars and their contents
    pillars = canonical_data.get("pillars", [])
    total_cv = 0
    total_sv = 0
    total_evidence = 0

    for pillar_data in pillars:
        await load_pillar(session, pillar_data, source_doc_id, run_id)

        for cv_data in pillar_data.get("core_values", []):
            await load_core_value(
                session, cv_data, pillar_data["id"], source_doc_id, run_id
            )
            total_cv += 1

            for sv_data in cv_data.get("sub_values", []):
                await load_sub_value(
                    session, sv_data, cv_data["id"], source_doc_id, run_id
                )
                total_sv += 1

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
    }

