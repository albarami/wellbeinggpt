"""Scholar reasoning integration (deep mode).

These tests ensure deep-mode:
- produces the required section headings
- returns citations when evidence exists
- abstains with not_found=True if it cannot meet depth floor

Requires DB.
"""

from __future__ import annotations

import pytest


@pytest.mark.asyncio
async def test_deep_mode_produces_structured_answer(require_db) -> None:
    from sqlalchemy import text

    from apps.api.core.database import get_session
    from apps.api.core.scholar_reasoning import ScholarReasoner
    from apps.api.retrieve.hybrid_retriever import HybridRetriever, RetrievalInputs

    async with get_session() as session:
        # Pick an entity name to form a deep-ish question.
        row = (
            await session.execute(
                text("SELECT name_ar, id FROM core_value WHERE name_ar IS NOT NULL LIMIT 1")
            )
        ).fetchone()
        if not row:
            pytest.skip("No core_value found")

        q = f"ما العلاقة بين {row.name_ar} وبين ركيزة أخرى داخل الإطار؟"

        retriever = HybridRetriever()
        retriever._session = session  # type: ignore[attr-defined]

        # Build a minimal evidence set via retriever.
        merged = await retriever.retrieve(
            session,
            inputs=RetrievalInputs(query=q, resolved_entities=[], top_k=10, graph_depth=2),
        )

        reasoner = ScholarReasoner(session=session, retriever=retriever)
        res = await reasoner.generate(
            question=q,
            question_norm=q,
            detected_entities=[],
            evidence_packets=merged.evidence_packets,
        )

        # Either it answers with structure + citations or abstains safely.
        if res.get("not_found"):
            assert (res.get("answer_ar") or "").strip()
            assert res.get("citations") == []
            return

        ans = str(res.get("answer_ar") or "")
        assert "تعريف المفهوم داخل الإطار" in ans
        assert "التأصيل والأدلة" in ans
        assert "الربط بين الركائز" in ans
        assert "خلاصة تنفيذية" in ans
        assert res.get("citations") and len(res.get("citations")) >= 1
