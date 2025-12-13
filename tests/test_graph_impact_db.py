import os

import pytest
from sqlalchemy import text

from apps.api.core.database import get_session
from apps.api.graph.impact import impact_propagation


@pytest.mark.asyncio
async def test_impact_propagation_scores_shared_refs_higher():
    """
    Expected: entities sharing more refs score higher.
    """
    if not os.getenv("DATABASE_URL") or os.getenv("RUN_DB_TESTS") != "1":
        pytest.skip("Requires DATABASE_URL and RUN_DB_TESTS=1")

    r1 = "quran:البقرة:777"
    r2 = "quran:البقرة:778"
    async with get_session() as session:
        try:
            await session.execute(
                text(
                    """
                    INSERT INTO edge (from_type, from_id, rel_type, to_type, to_id,
                                      created_method, created_by, justification, status)
                    VALUES
                        ('sub_value','SV_SEED','MENTIONS_REF','ref',CAST(:r1 AS VARCHAR(50)),'rule_exact_match','test',:r1,'approved'),
                        ('sub_value','SV_SEED','MENTIONS_REF','ref',CAST(:r2 AS VARCHAR(50)),'rule_exact_match','test',:r2,'approved'),
                        ('sub_value','SV_M1','MENTIONS_REF','ref',CAST(:r1 AS VARCHAR(50)),'rule_exact_match','test',:r1,'approved'),
                        ('sub_value','SV_M2','MENTIONS_REF','ref',CAST(:r1 AS VARCHAR(50)),'rule_exact_match','test',:r1,'approved'),
                        ('sub_value','SV_M2','MENTIONS_REF','ref',CAST(:r2 AS VARCHAR(50)),'rule_exact_match','test',:r2,'approved')
                    ON CONFLICT DO NOTHING
                    """
                ),
                {"r1": r1, "r2": r2},
            )
            await session.commit()

            data = await impact_propagation(session, entity_type="sub_value", entity_id="SV_SEED", max_depth=2, top_k=10)
            ids = [i["entity_id"] for i in data["items"]]
            assert "SV_M2" in ids and "SV_M1" in ids
            # SV_M2 shares 2 refs; SV_M1 shares 1 ref -> SV_M2 should rank ahead
            assert ids.index("SV_M2") < ids.index("SV_M1")
        finally:
            await session.execute(
                text(
                    """
                    DELETE FROM edge
                    WHERE created_by='test'
                      AND rel_type='MENTIONS_REF'
                      AND to_type='ref'
                      AND to_id IN (:r1, :r2)
                      AND from_id IN ('SV_SEED','SV_M1','SV_M2')
                    """
                ),
                {"r1": r1, "r2": r2},
            )
            await session.commit()


