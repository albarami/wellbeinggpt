"""Scholar Notes Pack ingestion.

Loads `data/scholar_notes/notes_v*.jsonl` into the database:
- Inserts note-derived chunks into `chunk` for retrieval.
- Inserts semantic edges into `edge` using `relation_type`.
- Inserts edge-level grounding rows into `edge_justification_span`.

Non-negotiable gates:
- Any evidence span must point to an existing chunk and valid offsets.
- Any evidence span source_id must be in source_inventory.
- Any semantic edge must have >=1 edge_justification_span.

All logic is deterministic (no LLM).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from apps.api.ingest.chunk_span_store import populate_chunk_spans_for_source
from apps.api.ingest.loader import create_ingestion_run, create_source_document, complete_ingestion_run
from apps.api.ingest.scholar_notes_schema import (
    ScholarEvidenceSpan,
    ScholarNoteRow,
    ScholarRelationType,
    primary_entity_for_note,
)
from apps.api.ingest.source_inventory import SourceInventory, build_source_inventory


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _token_count_estimate(text_ar: str) -> int:
    # Reason: deterministic and cheap; not a true tokenizer.
    return len([t for t in (text_ar or "").split() if t.strip()])


def _stable_chunk_id(*parts: str, max_len: int = 50) -> str:
    """Create a stable chunk_id <= max_len."""

    raw = "|".join([p for p in parts if p is not None])
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]
    cid = f"SN_{digest}"
    return cid[:max_len]


@dataclass(frozen=True)
class ScholarNotesIngestStats:
    inserted_chunks: int
    inserted_edges: int
    inserted_edge_spans: int
    inserted_chunk_spans: int


async def _load_rows_jsonl(path: Path) -> list[ScholarNoteRow]:
    rows: list[ScholarNoteRow] = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = (line or "").strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception as e:
                raise ValueError(f"Invalid JSON at line {i}: {e}")
            rows.append(ScholarNoteRow.model_validate(obj))
    return rows


async def _fetch_chunks_text(session: AsyncSession, chunk_ids: list[str]) -> dict[str, dict[str, Any]]:
    if not chunk_ids:
        return {}

    rows = (
        await session.execute(
            text(
                """
                SELECT chunk_id, text_ar, source_doc_id
                FROM chunk
                WHERE chunk_id = ANY(:cids)
                """
            ),
            {"cids": chunk_ids},
        )
    ).fetchall()

    out: dict[str, dict[str, Any]] = {}
    for r in rows:
        out[str(r.chunk_id)] = {
            "text_ar": str(r.text_ar or ""),
            "source_doc_id": str(r.source_doc_id or ""),
        }
    return out


def _select_edge_justification_spans(
    *,
    link_justification_ar: str,
    evidence_spans: list[ScholarEvidenceSpan],
) -> list[ScholarEvidenceSpan]:
    """Select evidence spans that ground a specific cross-pillar link.

    Deterministic rule:
    - Pick spans whose `quote` appears verbatim in `justification_ar`.

    This forces `justification_ar` to remain bindable and avoids ambiguous mapping.
    """

    just = (link_justification_ar or "").strip()
    if not just:
        return []

    chosen: list[ScholarEvidenceSpan] = []
    for sp in evidence_spans:
        q = (sp.quote or "").strip()
        if q and q in just:
            chosen.append(sp)
    return chosen


async def _validate_spans_and_inventory(
    *,
    session: AsyncSession,
    inventory: SourceInventory,
    spans: list[ScholarEvidenceSpan],
) -> None:
    """Validate evidence spans against DB + source inventory."""

    # 1) Inventory gate
    for sp in spans:
        inventory.assert_allowed(str(sp.source_id))

    # 2) Chunk existence + offsets + quote check + source_id match
    uniq = sorted({str(s.chunk_id) for s in spans if str(s.chunk_id)})
    chunk_map = await _fetch_chunks_text(session, uniq)

    missing = [cid for cid in uniq if cid not in chunk_map]
    if missing:
        raise ValueError(f"Evidence spans reference missing chunk_ids: {missing[:10]}")

    for sp in spans:
        cid = str(sp.chunk_id)
        meta = chunk_map.get(cid) or {}
        txt = str(meta.get("text_ar") or "")
        if not txt:
            raise ValueError(f"Evidence span references empty chunk text: {cid}")

        if str(meta.get("source_doc_id") or "") != str(sp.source_id):
            raise ValueError(
                f"Evidence span source_id mismatch for {cid}: span.source_id={sp.source_id} chunk.source_doc_id={meta.get('source_doc_id')}"
            )

        s = int(sp.span_start)
        e = int(sp.span_end)
        if e <= s:
            raise ValueError(f"Invalid span offsets for {cid}: start={s} end={e}")
        if s < 0 or e > len(txt):
            raise ValueError(f"Span offsets out of range for {cid}: start={s} end={e} len={len(txt)}")

        extracted = txt[s:e]
        q = str(sp.quote or "").strip()
        if not q:
            raise ValueError(f"Empty quote for evidence span {cid}:{s}-{e}")
        if q not in extracted:
            # Allow whitespace-trim mismatch (common when writers quote a subset)
            if q.strip() not in extracted.strip():
                raise ValueError(f"Quote not found in extracted span for {cid}:{s}-{e}")


async def ingest_scholar_notes_jsonl(
    *,
    session: AsyncSession,
    notes_jsonl_path: str,
    pack_name: str = "scholar_notes_v1",
) -> dict[str, Any]:
    """Ingest scholar notes JSONL into DB.

    Args:
        session: DB session.
        notes_jsonl_path: Path to notes JSONL.
        pack_name: Logical pack name for provenance.

    Returns:
        Summary dict.
    """

    path = Path(notes_jsonl_path)
    if not path.exists():
        raise FileNotFoundError(str(path))

    file_hash = _sha256_file(path)

    # Create a source document row for this pack.
    source_doc_id = await create_source_document(
        session=session,
        file_name=str(path.name),
        file_hash=file_hash,
        framework_version=pack_name,
    )
    run_id = await create_ingestion_run(session=session, source_doc_id=source_doc_id)

    inventory = await build_source_inventory(session)

    inserted_chunks = 0
    inserted_edges = 0
    inserted_edge_spans = 0

    try:
        rows = await _load_rows_jsonl(path)

        # Validate ALL referenced evidence spans before inserting anything.
        all_spans: list[ScholarEvidenceSpan] = []
        for r in rows:
            all_spans.extend(list(r.evidence_spans or []))
        await _validate_spans_and_inventory(session=session, inventory=inventory, spans=all_spans)

        # Insert note chunks (attached to the primary entity of the note).
        for r in rows:
            ent_type, ent_id = primary_entity_for_note(r)

            # One chunk per major section (definition/commentary). Empty sections are allowed.
            chunks_to_insert: list[tuple[str, str, str]] = []  # (chunk_type, text, section)
            if (r.definition_ar or "").strip():
                chunks_to_insert.append(("definition", str(r.definition_ar).strip(), "definition"))
            if (r.deep_explanation_ar or "").strip():
                chunks_to_insert.append(("commentary", str(r.deep_explanation_ar).strip(), "deep_explanation"))

            # Optional scenario/misunderstanding chunks.
            for i, sc in enumerate(r.applied_scenarios or [], start=1):
                chunks_to_insert.append(("commentary", f"سيناريو: {sc.scenario_ar}\n\nتحليل: {sc.analysis_ar}", f"scenario_{i}"))

            for i, m in enumerate(r.common_misunderstandings or [], start=1):
                chunks_to_insert.append(("commentary", f"سوء فهم شائع: {m.misunderstanding_ar}\n\nالتصحيح: {m.correction_ar}", f"misunderstanding_{i}"))

            for chunk_type, text_ar, section in chunks_to_insert:
                cid = _stable_chunk_id(pack_name, r.version, r.note_id, section, ent_type, ent_id, chunk_type)
                source_anchor = json.dumps(
                    {
                        "pack": pack_name,
                        "version": r.version,
                        "note_id": r.note_id,
                        "section": section,
                    },
                    ensure_ascii=False,
                )

                await session.execute(
                    text(
                        """
                        INSERT INTO chunk (
                          chunk_id, entity_type, entity_id, chunk_type, text_ar, text_en,
                          source_doc_id, source_anchor, token_count_estimate
                        )
                        VALUES (
                          :chunk_id, :entity_type, :entity_id, :chunk_type, :text_ar, NULL,
                          :source_doc_id, :source_anchor, :toks
                        )
                        ON CONFLICT (chunk_id) DO UPDATE SET
                          text_ar = EXCLUDED.text_ar,
                          source_anchor = EXCLUDED.source_anchor,
                          source_doc_id = EXCLUDED.source_doc_id,
                          token_count_estimate = EXCLUDED.token_count_estimate
                        """
                    ),
                    {
                        "chunk_id": cid,
                        "entity_type": ent_type,
                        "entity_id": ent_id,
                        "chunk_type": chunk_type,
                        "text_ar": text_ar,
                        "source_doc_id": source_doc_id,
                        "source_anchor": source_anchor,
                        "toks": int(_token_count_estimate(text_ar)),
                    },
                )
                inserted_chunks += 1

            # Insert semantic edges (if any).
            for link in r.cross_pillar_links or []:
                to_type = "sub_value" if link.target_sub_value_id else "pillar"
                to_id = str(link.target_sub_value_id or link.target_pillar_id)

                # Select evidence spans that ground this link.
                chosen = _select_edge_justification_spans(
                    link_justification_ar=str(link.justification_ar),
                    evidence_spans=list(r.evidence_spans or []),
                )
                if not chosen:
                    raise ValueError(
                        f"No justification evidence spans found for link in note_id={r.note_id}. "
                        "Rule: at least one evidence span quote must appear verbatim in justification_ar."
                    )

                strength = min(1.0, 0.2 * float(len(chosen)))

                edge_row = (
                    await session.execute(
                        text(
                            """
                            INSERT INTO edge (
                              from_type, from_id, rel_type, relation_type,
                              to_type, to_id,
                              created_method, created_by, justification,
                              strength_score, status
                            )
                            VALUES (
                              :ft, :fid, 'SCHOLAR_LINK', :relation_type,
                              :tt, :tid,
                              'human_approved', :created_by, :justification,
                              :strength_score, 'approved'
                            )
                            ON CONFLICT DO NOTHING
                            RETURNING id
                            """
                        ),
                        {
                            "ft": ent_type,
                            "fid": ent_id,
                            "relation_type": str(link.relation_type.value),
                            "tt": to_type,
                            "tid": to_id,
                            "created_by": pack_name,
                            "justification": f"note_id={r.note_id}",
                            "strength_score": float(strength),
                        },
                    )
                ).fetchone()

                edge_id = str(edge_row.id) if edge_row and getattr(edge_row, "id", None) else None
                if not edge_id:
                    # Edge already existed; we still must ensure it has spans.
                    # Fetch it and attach spans if missing.
                    existing = (
                        await session.execute(
                            text(
                                """
                                SELECT id
                                FROM edge
                                WHERE from_type=:ft AND from_id=:fid
                                  AND rel_type='SCHOLAR_LINK' AND relation_type=:rt
                                  AND to_type=:tt AND to_id=:tid
                                LIMIT 1
                                """
                            ),
                            {
                                "ft": ent_type,
                                "fid": ent_id,
                                "rt": str(link.relation_type.value),
                                "tt": to_type,
                                "tid": to_id,
                            },
                        )
                    ).fetchone()
                    edge_id = str(existing.id) if existing and getattr(existing, "id", None) else None

                if not edge_id:
                    raise ValueError("Failed to resolve inserted/existing edge_id")

                inserted_edges += 1

                # Insert edge justification spans (multi-span supported).
                for sp in chosen:
                    await session.execute(
                        text(
                            """
                            INSERT INTO edge_justification_span (edge_id, chunk_id, span_start, span_end, quote)
                            VALUES (:edge_id, :chunk_id, :s, :e, :q)
                            ON CONFLICT DO NOTHING
                            """
                        ),
                        {
                            "edge_id": edge_id,
                            "chunk_id": str(sp.chunk_id),
                            "s": int(sp.span_start),
                            "e": int(sp.span_end),
                            "q": str(sp.quote),
                        },
                    )
                    inserted_edge_spans += 1

                # Hard gate: verify spans exist for this edge.
                row_cnt = (
                    await session.execute(
                        text(
                            """
                            SELECT COUNT(*) AS c
                            FROM edge_justification_span
                            WHERE edge_id=:edge_id
                            """
                        ),
                        {"edge_id": edge_id},
                    )
                ).fetchone()
                if not row_cnt or int(getattr(row_cnt, "c", 0) or 0) <= 0:
                    raise ValueError(f"Hard gate failed: edge has no justification spans (edge_id={edge_id})")

        # Populate deterministic sentence spans for all inserted note chunks (for eval + binding).
        inserted_chunk_spans = await populate_chunk_spans_for_source(session, source_doc_id)

        await complete_ingestion_run(
            session=session,
            run_id=run_id,
            entities_extracted=len(rows),
            evidence_extracted=int(inserted_edge_spans),
            validation_errors=[],
            status="completed",
        )

        return {
            "source_doc_id": source_doc_id,
            "run_id": run_id,
            "notes": len(rows),
            "chunks": inserted_chunks,
            "edges": inserted_edges,
            "edge_spans": inserted_edge_spans,
            "chunk_spans": inserted_chunk_spans,
        }

    except Exception as e:
        # Mark run failed but re-raise.
        # Reason: Postgres aborts the current transaction after an error, so we must
        # rollback before attempting to write failure status.
        try:
            await session.rollback()
        except Exception:
            pass
        await complete_ingestion_run(
            session=session,
            run_id=run_id,
            entities_extracted=0,
            evidence_extracted=0,
            validation_errors=[str(e)],
            status="failed",
        )
        raise
