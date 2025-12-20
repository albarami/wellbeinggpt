"""Ingestion loader: canonical JSON pipeline.

Reason: keep each module <500 LOC.
"""

from __future__ import annotations

import os
from typing import Any

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from apps.api.ingest.loader_chunks import embed_all_chunks_for_source, load_chunks_jsonl
from apps.api.ingest.loader_edges import build_edges_for_source
from apps.api.ingest.loader_entities import load_core_value, load_pillar, load_sub_value
from apps.api.ingest.loader_meta import complete_ingestion_run, create_ingestion_run, create_source_document
from apps.api.ingest.loader_text import load_evidence, load_text_block
from apps.api.graph.framework_edge_miner import extract_semantic_edges_from_chunk, upsert_mined_edges


async def load_canonical_json_to_db(session: AsyncSession, canonical_data: dict[str, Any], file_name: str) -> dict[str, Any]:
    """Load canonical JSON data into the database."""
    meta = canonical_data.get("meta", {})
    file_hash = meta.get("source_file_hash", "unknown")
    framework_version = meta.get("framework_version", "unknown")

    source_doc_id = await create_source_document(session, file_name, file_hash, framework_version)

    purge = os.getenv("INGEST_PURGE_EXISTING", "true").lower() in ("1", "true", "yes")
    if purge:
        for stmt in [
            # IMPORTANT: remove any edge justifications that reference chunks in this source.
            # Reason: edge_justification_span.chunk_id has an FK to chunk; leaving these rows can
            # break chunk deletion even when the edge itself is not tied to this source (e.g.,
            # argument-grade edges created in tests).
            "DELETE FROM edge_justification_span WHERE chunk_id IN (SELECT chunk_id FROM chunk WHERE source_doc_id=:sd)",
            # Argument-grade evidence spans may also reference chunks.
            "DELETE FROM argument_evidence_span WHERE chunk_id IN (SELECT chunk_id FROM chunk WHERE source_doc_id=:sd)",
            "DELETE FROM edge WHERE from_id IN (SELECT id FROM pillar WHERE source_doc_id=:sd) OR to_id IN (SELECT id FROM pillar WHERE source_doc_id=:sd)",
            "DELETE FROM edge WHERE from_id IN (SELECT id FROM core_value WHERE source_doc_id=:sd) OR to_id IN (SELECT id FROM core_value WHERE source_doc_id=:sd)",
            "DELETE FROM edge WHERE from_id IN (SELECT id FROM sub_value WHERE source_doc_id=:sd) OR to_id IN (SELECT id FROM sub_value WHERE source_doc_id=:sd)",
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
                # Rollback so the session isn't left in an aborted transaction state.
                try:
                    await session.rollback()
                except Exception:
                    pass
                continue

        try:
            await session.execute(
                text(
                    """
                    DELETE FROM edge e
                    WHERE e.status = 'approved'
                      AND (
                        (e.from_type='pillar' AND NOT EXISTS (SELECT 1 FROM pillar p WHERE p.id=e.from_id)) OR
                        (e.from_type='core_value' AND NOT EXISTS (SELECT 1 FROM core_value cv WHERE cv.id=e.from_id)) OR
                        (e.from_type='sub_value' AND NOT EXISTS (SELECT 1 FROM sub_value sv WHERE sv.id=e.from_id)) OR
                        (e.from_type='chunk' AND NOT EXISTS (SELECT 1 FROM chunk ch WHERE ch.chunk_id=e.from_id)) OR
                        (e.from_type='evidence' AND NOT EXISTS (SELECT 1 FROM evidence ev WHERE ev.id::text=e.from_id)) OR
                        (e.to_type='pillar' AND NOT EXISTS (SELECT 1 FROM pillar p2 WHERE p2.id=e.to_id)) OR
                        (e.to_type='core_value' AND NOT EXISTS (SELECT 1 FROM core_value cv2 WHERE cv2.id=e.to_id)) OR
                        (e.to_type='sub_value' AND NOT EXISTS (SELECT 1 FROM sub_value sv2 WHERE sv2.id=e.to_id)) OR
                        (e.to_type='chunk' AND NOT EXISTS (SELECT 1 FROM chunk ch2 WHERE ch2.chunk_id=e.to_id)) OR
                        (e.to_type='evidence' AND NOT EXISTS (SELECT 1 FROM evidence ev2 WHERE ev2.id::text=e.to_id))
                      )
                    """
                )
            )
        except Exception:
            pass

    # Create the ingestion run AFTER purge operations.
    # Reason: purge may rollback the session on best-effort delete failures.
    run_id = await create_ingestion_run(session, source_doc_id)

    pillars = canonical_data.get("pillars", [])
    total_cv = 0
    total_sv = 0
    total_evidence = 0

    pillar_id_map: dict[str, str] = {}
    core_id_map: dict[str, str] = {}
    sub_id_map: dict[str, str] = {}

    for pillar_data in pillars:
        pillar_db_id = await load_pillar(session, pillar_data, source_doc_id, run_id)
        pillar_id_map[str(pillar_data["id"])] = str(pillar_db_id)

        for cv_data in pillar_data.get("core_values", []):
            cv_db_id = await load_core_value(session, cv_data, pillar_db_id, source_doc_id, run_id)
            core_id_map[str(cv_data["id"])] = str(cv_db_id)
            total_cv += 1

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
                sv_db_id = await load_sub_value(session, sv_data, cv_db_id, source_doc_id, run_id)
                sub_id_map[str(sv_data["id"])] = str(sv_db_id)
                total_sv += 1

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

    for b in canonical_data.get("supplemental_text_blocks", []) or []:
        try:
            et = str(b.get("entity_type") or "")
            eid = str(b.get("entity_id") or "")
            bt = str(b.get("block_type") or "supplemental_ocr")
            text_ar = str(b.get("text_ar") or "").strip()
            source_anchor = b.get("source_anchor") or {}
            if not et or not eid or not text_ar:
                continue

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
        try:
            total_embeddings = await embed_all_chunks_for_source(session=session, source_doc_id=source_doc_id)
        except Exception:
            total_embeddings = 0

    await build_edges_for_source(session=session, source_doc_id=source_doc_id)

    # Framework-only: mine explicit cross-pillar semantic edges from chunk text.
    # Reason: the framework itself contains grounded cross-pillar claims that should become
    # SCHOLAR_LINK edges with justification spans (enables stakeholder cross-pillar reasoning).
    try:
        enable_miner = os.getenv("ENABLE_FRAMEWORK_EDGE_MINER", "1").lower() in ("1", "true", "yes", "on")
        if enable_miner and ("framework" in str(file_name or "").lower()):
            # Pull all chunks for this source and mine sentence-level edges.
            rows = (
                await session.execute(
                    text(
                        """
                        SELECT chunk_id, text_ar
                        FROM chunk
                        WHERE source_doc_id::text = :sd
                        ORDER BY chunk_id
                        """
                    ),
                    {"sd": str(source_doc_id)},
                )
            ).fetchall()
            mined = []
            for r in rows:
                mined.extend(extract_semantic_edges_from_chunk(chunk_id=str(r.chunk_id), text_ar=str(r.text_ar or "")))
            mined = [e for e in mined if e.from_pillar_id != e.to_pillar_id]
            if mined:
                await upsert_mined_edges(
                    session=session,
                    mined=mined,
                    created_by="framework_semantic_edge_miner",
                    strength_score=0.8,
                )
    except Exception:
        # Fail-open: do not block framework ingestion if miner fails; system remains safe (no edges).
        pass

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

