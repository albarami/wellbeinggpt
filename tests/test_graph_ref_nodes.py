import os

import pytest

from apps.api.core.database import get_session
from apps.api.core.schemas import EntityType
from apps.api.retrieve.graph_retriever import expand_graph
from sqlalchemy import text


@pytest.mark.asyncio
async def test_expand_graph_traverses_ref_nodes():
    """
    The graph must support ref nodes (Quran/Hadith refs) as first-class nodes.

    We validate the multi-hop traversal:
      sub_value(SV_A) --MENTIONS_REF--> ref(quran:البقرة:1) <--MENTIONS_REF-- sub_value(SV_B)

    This enables discovery of other entities that share the same verse/hadith.
    """
    if not os.getenv("DATABASE_URL") or os.getenv("RUN_DB_TESTS") != "1":
        pytest.skip("Requires DATABASE_URL and RUN_DB_TESTS=1")

    async with get_session() as session:
        # Clean any residue for deterministic test reruns
        await session.execute(
            text(
                """
                DELETE FROM edge
                WHERE (from_type, from_id, rel_type, to_type, to_id) IN (
                    ('sub_value','SV_A','MENTIONS_REF','ref','quran:البقرة:1'),
                    ('sub_value','SV_B','MENTIONS_REF','ref','quran:البقرة:1')
                )
                """
            )
        )
        await session.commit()

        await session.execute(
            text(
                """
                INSERT INTO edge (from_type, from_id, rel_type, to_type, to_id,
                                  created_method, created_by, justification, status)
                VALUES
                    ('sub_value','SV_A','MENTIONS_REF','ref','quran:البقرة:1',
                     'rule_exact_match','test','quran:البقرة:1','approved'),
                    ('sub_value','SV_B','MENTIONS_REF','ref','quran:البقرة:1',
                     'rule_exact_match','test','quran:البقرة:1','approved')
                ON CONFLICT DO NOTHING
                """
            )
        )
        await session.commit()

        reached = await expand_graph(
            session,
            EntityType.SUB_VALUE,
            "SV_A",
            depth=2,
            relationship_types=["MENTIONS_REF"],
        )

        # We should reach the ref node (depth=1) and SV_B (depth=2).
        assert any(r["neighbor_type"] == "ref" and r["neighbor_id"] == "quran:البقرة:1" for r in reached)
        assert any(r["neighbor_type"] == "sub_value" and r["neighbor_id"] == "SV_B" for r in reached)


