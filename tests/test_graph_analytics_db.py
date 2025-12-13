import os

import pytest
from sqlalchemy import text

from apps.api.core.database import get_session
from apps.api.graph.analytics import top_ref_nodes


@pytest.mark.asyncio
async def test_top_ref_nodes_ranks_by_entity_count():
    """
    Expected: ranking orders ref nodes by how many distinct entities mention them.
    Edge case: evidence_type filter limits to the requested type.
    """
    if not os.getenv("DATABASE_URL") or os.getenv("RUN_DB_TESTS") != "1":
        pytest.skip("Requires DATABASE_URL and RUN_DB_TESTS=1")

    async with get_session() as session:
        # Insert a tiny synthetic subgraph using edges only (no need to create full entities).
        await session.execute(
            text(
                """
                INSERT INTO edge (from_type, from_id, rel_type, to_type, to_id,
                                  created_method, created_by, justification, status)
                VALUES
                    ('sub_value','SV_X','MENTIONS_REF','ref','quran:البقرة:1','rule_exact_match','test','quran:البقرة:1','approved'),
                    ('sub_value','SV_Y','MENTIONS_REF','ref','quran:البقرة:1','rule_exact_match','test','quran:البقرة:1','approved'),
                    ('sub_value','SV_Z','MENTIONS_REF','ref','quran:البقرة:2','rule_exact_match','test','quran:البقرة:2','approved'),
                    ('sub_value','SV_Z','MENTIONS_REF','ref','hadith:muslim:1','rule_exact_match','test','hadith:muslim:1','approved')
                ON CONFLICT DO NOTHING
                """
            )
        )
        await session.commit()

        items = await top_ref_nodes(session, evidence_type="quran", created_by="test", top_k=10)
        assert items[0]["ref_node_id"] == "quran:البقرة:1"
        assert items[0]["entity_count"] >= 2

        # Filter: hadith should not appear
        assert all(i["ref_node_id"].startswith("quran:") for i in items)


