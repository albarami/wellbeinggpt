import os

import pytest
from sqlalchemy import text

from apps.api.core.database import get_session
from apps.api.graph.explain import shortest_path


@pytest.mark.asyncio
async def test_shortest_path_finds_ref_bridge():
    """
    Expected: path exists via ref node.
    Failure: if no edges exist, found=false.
    """
    if not os.getenv("DATABASE_URL") or os.getenv("RUN_DB_TESTS") != "1":
        pytest.skip("Requires DATABASE_URL and RUN_DB_TESTS=1")

    rid = "quran:البقرة:888"
    async with get_session() as session:
        try:
            await session.execute(
                text(
                    """
                    INSERT INTO edge (from_type, from_id, rel_type, to_type, to_id,
                                      created_method, created_by, justification, status)
                    VALUES
                        ('sub_value','SV_A2','MENTIONS_REF','ref',CAST(:rid AS VARCHAR(50)),'rule_exact_match','test',:rid,'approved'),
                        ('sub_value','SV_B2','MENTIONS_REF','ref',CAST(:rid AS VARCHAR(50)),'rule_exact_match','test',:rid,'approved')
                    ON CONFLICT DO NOTHING
                    """
                ),
                {"rid": rid},
            )
            await session.commit()

            res = await shortest_path(
                session,
                start_type="sub_value",
                start_id="SV_A2",
                target_type="sub_value",
                target_id="SV_B2",
                max_depth=3,
                rel_types=["MENTIONS_REF"],
            )
            assert res["found"] is True
            # Expect SV_A2 -> ref -> SV_B2
            types = [n["type"] for n in res["path"]]
            assert types[0] == "sub_value"
            assert "ref" in types
            assert types[-1] == "sub_value"
        finally:
            await session.execute(
                text(
                    """
                    DELETE FROM edge
                    WHERE created_by='test'
                      AND rel_type='MENTIONS_REF'
                      AND to_type='ref'
                      AND to_id=:rid
                      AND from_id IN ('SV_A2','SV_B2')
                    """
                ),
                {"rid": rid},
            )
            await session.commit()


