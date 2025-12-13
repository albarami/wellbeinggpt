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
        try:
            # Insert a tiny synthetic subgraph using edges only.
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
        finally:
            # Cleanup synthetic edges so global DB integrity tests don't see orphans
            await session.execute(
                text(
                    """
                    DELETE FROM edge
                    WHERE created_by='test'
                      AND rel_type='MENTIONS_REF'
                      AND to_type='ref'
                      AND to_id IN ('quran:البقرة:1','quran:البقرة:2','hadith:muslim:1')
                      AND from_id IN ('SV_X','SV_Y','SV_Z')
                    """
                )
            )
            await session.commit()


