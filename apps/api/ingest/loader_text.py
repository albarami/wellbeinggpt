"""Ingestion loader: text blocks + evidence records.

Reason: keep each module <500 LOC.
"""

from __future__ import annotations

import json
import uuid
from typing import Optional

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from apps.api.ingest.loader_meta import _scalar_one_or_none


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
    """Load an anchored text block (definition/commentary/supplemental_ocr)."""
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
            {"id": existing_id, "text_ar": text_ar, "source_anchor": json.dumps(source_anchor), "run_id": run_id},
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
    """Load an evidence record into the database (deduped by anchor + content)."""
    anchor_str = ""
    try:
        if isinstance(source_anchor, dict):
            anchor_str = str(source_anchor.get("source_anchor") or "")
    except Exception:
        anchor_str = ""

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
          AND (source_anchor->>'source_anchor') = :a
        """,
        {
            "sd": source_doc_id,
            "et": entity_type,
            "eid": entity_id,
            "ev": evidence_type,
            "ref_norm": ref_norm or "",
            "text_ar": text_ar,
            "a": anchor_str,
        },
    )
    if existing_id:
        return existing_id

    if (parse_status or "").lower() != "success":
        ref_norm = None
        surah_name_ar = None
        surah_number = None
        ayah_number = None
        hadith_collection = None
        hadith_number = None
    else:
        if ref_norm and len(ref_norm) > 255:
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
        text(
            """
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
            """
        ),
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
        },
    )
    return evidence_id

