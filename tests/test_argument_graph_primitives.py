from __future__ import annotations

import pytest


@pytest.mark.asyncio
async def test_argument_claim_supported_by_span_is_grounded(require_db) -> None:
    """
    Ensure new argument primitives can be created and are grounded via edge_justification_span.
    """

    from sqlalchemy import text

    from apps.api.core.database import get_session
    from apps.api.graph.argument_store import create_claim, create_evidence_span, link_claim_supported_by_span

    async with get_session() as session:
        # Need an existing chunk to reference for the evidence span.
        row = (await session.execute(text("SELECT chunk_id, text_ar FROM chunk ORDER BY chunk_id LIMIT 1"))).fetchone()
        assert row is not None
        chunk_id = str(row.chunk_id)
        txt = str(row.text_ar or "")
        quote = (txt[:40] or "q").strip()

        cid = await create_claim(session=session, text_ar="ادعاء اختباري")
        sid = await create_evidence_span(session=session, chunk_id=chunk_id, span_start=0, span_end=min(40, len(txt)), quote=quote)
        eid = await link_claim_supported_by_span(
            session=session,
            claim_id=cid,
            span_id=sid,
            span_chunk_id=chunk_id,
            span_start=0,
            span_end=min(40, len(txt)),
            quote=quote,
        )
        await session.commit()

        # Verify edge exists and has at least one justification span row.
        e = (
            await session.execute(
                text(
                    """
                    SELECT id::text, rel_type, relation_type
                    FROM edge
                    WHERE id::text = :id
                    """
                ),
                {"id": eid},
            )
        ).fetchone()
        assert e is not None
        assert str(e.rel_type) == "ARG_LINK"
        assert str(e.relation_type) == "SUPPORTED_BY"

        js = (
            await session.execute(
                text("SELECT COUNT(*) FROM edge_justification_span WHERE edge_id::text = :id"),
                {"id": eid},
            )
        ).scalar_one()
        assert int(js or 0) >= 1

