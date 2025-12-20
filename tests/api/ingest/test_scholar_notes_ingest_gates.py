"""Scholar notes ingestion gates (DB integration).

Covers expected + failure + edge cases for the hard gates:
- evidence spans must reference existing chunks with valid offsets
- source_inventory must contain span.source_id
- semantic edge must have >= 1 justification span (selected deterministically)

These tests require a live DB (`require_db`).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest


@pytest.mark.asyncio
async def test_ingest_rejects_missing_chunk(require_db, tmp_path: Path) -> None:
    from apps.api.core.database import get_session
    from apps.api.ingest.scholar_notes_loader import ingest_scholar_notes_jsonl

    notes_path = tmp_path / "notes_v1.jsonl"

    # Choose a chunk_id that almost certainly doesn't exist.
    row = {
        "note_id": "sn-missing-chunk",
        "pillar_id": "P001",
        "core_value_id": None,
        "sub_value_id": None,
        "title_ar": "اختبار: مفقود",
        "definition_ar": "تعريف قصير",
        "deep_explanation_ar": "",
        "cross_pillar_links": [],
        "applied_scenarios": [],
        "common_misunderstandings": [],
        "evidence_spans": [
            {
                "source_id": "NOT_A_REAL_SOURCE_ID",
                "chunk_id": "CH_DOES_NOT_EXIST",
                "span_start": 0,
                "span_end": 4,
                "quote": "نص",
            }
        ],
        "tags": ["اختبار"],
        "version": "v1",
    }
    notes_path.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")

    async with get_session() as session:
        with pytest.raises(Exception):
            await ingest_scholar_notes_jsonl(session=session, notes_jsonl_path=str(notes_path))


@pytest.mark.asyncio
async def test_ingest_rejects_link_without_quote_binding(require_db, tmp_path: Path) -> None:
    from sqlalchemy import text

    from apps.api.core.database import get_session
    from apps.api.ingest.scholar_notes_loader import ingest_scholar_notes_jsonl

    # We need a real existing chunk span to pass chunk existence.
    async with get_session() as session:
        ch = (
            await session.execute(
                text(
                    """
                    SELECT chunk_id, source_doc_id, text_ar
                    FROM chunk
                    WHERE text_ar IS NOT NULL AND text_ar <> ''
                    LIMIT 1
                    """
                )
            )
        ).fetchone()

        assert ch and ch.chunk_id and ch.source_doc_id and ch.text_ar
        txt = str(ch.text_ar)
        quote = txt[: min(12, len(txt))].strip() or "نص"

        notes_path = tmp_path / "notes_v1.jsonl"
        row = {
            "note_id": "sn-link-no-bind",
            "pillar_id": "P001",
            "core_value_id": None,
            "sub_value_id": None,
            "title_ar": "اختبار: رابط غير مربوط",
            "definition_ar": "تعريف",
            "deep_explanation_ar": "",
            "cross_pillar_links": [
                {
                    "target_pillar_id": "P002",
                    "target_sub_value_id": None,
                    "relation_type": "COMPLEMENTS",
                    # Does NOT include quote => should fail deterministic selection.
                    "justification_ar": "تبرير لا يحتوي الاقتباس المطلوب",
                }
            ],
            "applied_scenarios": [],
            "common_misunderstandings": [],
            "evidence_spans": [
                {
                    "source_id": str(ch.source_doc_id),
                    "chunk_id": str(ch.chunk_id),
                    "span_start": 0,
                    "span_end": min(30, len(txt)),
                    "quote": quote,
                }
            ],
            "tags": ["اختبار"],
            "version": "v1",
        }
        notes_path.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")

        with pytest.raises(Exception):
            await ingest_scholar_notes_jsonl(session=session, notes_jsonl_path=str(notes_path))


@pytest.mark.asyncio
async def test_ingest_inserts_semantic_edge_with_spans(require_db, tmp_path: Path) -> None:
    from sqlalchemy import text

    from apps.api.core.database import get_session
    from apps.api.ingest.scholar_notes_loader import ingest_scholar_notes_jsonl

    async with get_session() as session:
        # Get a real chunk
        ch = (
            await session.execute(
                text(
                    """
                    SELECT chunk_id, source_doc_id, text_ar
                    FROM chunk
                    WHERE text_ar IS NOT NULL AND text_ar <> ''
                    LIMIT 1
                    """
                )
            )
        ).fetchone()
        assert ch and ch.chunk_id and ch.source_doc_id and ch.text_ar
        txt = str(ch.text_ar)

        # Use a small quote and ensure it appears in justification_ar
        quote = txt[: min(12, len(txt))].strip() or "نص"
        s = 0
        e = min(30, len(txt))

        notes_path = tmp_path / "notes_v1.jsonl"
        row = {
            "note_id": "sn-edge-ok",
            "pillar_id": "P001",
            "core_value_id": None,
            "sub_value_id": None,
            "title_ar": "اختبار: رابط صحيح",
            "definition_ar": "تعريف",
            "deep_explanation_ar": "",
            "cross_pillar_links": [
                {
                    "target_pillar_id": "P002",
                    "target_sub_value_id": None,
                    "relation_type": "REINFORCES",
                    "justification_ar": f"سبب الربط: {quote}",
                }
            ],
            "applied_scenarios": [],
            "common_misunderstandings": [],
            "evidence_spans": [
                {
                    "source_id": str(ch.source_doc_id),
                    "chunk_id": str(ch.chunk_id),
                    "span_start": s,
                    "span_end": e,
                    "quote": quote,
                }
            ],
            "tags": ["اختبار"],
            "version": "v1",
        }
        notes_path.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")

        out = await ingest_scholar_notes_jsonl(session=session, notes_jsonl_path=str(notes_path))
        assert out["edges"] >= 1
        assert out["edge_spans"] >= 1

        # Verify the semantic edge exists with at least one justification span.
        row_edge = (
            await session.execute(
                text(
                    """
                    SELECT e.id
                    FROM edge e
                    WHERE e.rel_type='SCHOLAR_LINK'
                      AND e.relation_type='REINFORCES'
                      AND e.status='approved'
                    ORDER BY e.created_at DESC
                    LIMIT 1
                    """
                )
            )
        ).fetchone()
        assert row_edge and row_edge.id

        row_span = (
            await session.execute(
                text(
                    """
                    SELECT 1
                    FROM edge_justification_span s
                    WHERE s.edge_id=:eid
                    LIMIT 1
                    """
                ),
                {"eid": str(row_edge.id)},
            )
        ).fetchone()
        assert row_span
