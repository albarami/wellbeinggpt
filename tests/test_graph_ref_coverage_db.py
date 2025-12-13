import os

import pytest
from sqlalchemy import text

from apps.api.core.database import get_session
from apps.api.graph.ref_coverage import ref_coverage


@pytest.mark.asyncio
async def test_ref_coverage_returns_entities_and_pillars_best_effort():
    """
    Expected: coverage includes entities that mention the ref.
    Edge: pillar list may be empty if entities aren't real rows (best-effort).
    """
    if not os.getenv("DATABASE_URL") or os.getenv("RUN_DB_TESTS") != "1":
        pytest.skip("Requires DATABASE_URL and RUN_DB_TESTS=1")

    rid = "quran:البقرة:999"
    async with get_session() as session:
        await session.execute(
            text(
                """
                INSERT INTO edge (from_type, from_id, rel_type, to_type, to_id,
                                  created_method, created_by, justification, status)
                VALUES
                    ('sub_value','SV_T1','MENTIONS_REF','ref',CAST(:rid AS VARCHAR(50)),'rule_exact_match','test',:rid,'approved'),
                    ('core_value','CV_T1','MENTIONS_REF','ref',CAST(:rid AS VARCHAR(50)),'rule_exact_match','test',:rid,'approved')
                ON CONFLICT DO NOTHING
                """
            ),
            {"rid": rid},
        )
        await session.commit()

        data = await ref_coverage(session, ref_node_id=rid, limit=50)
        assert data["ref_node_id"] == rid
        assert {"entity_type": "sub_value", "entity_id": "SV_T1"} in data["entities"]
        assert {"entity_type": "core_value", "entity_id": "CV_T1"} in data["entities"]
        assert isinstance(data["pillars"], list)


