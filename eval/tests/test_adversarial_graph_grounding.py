from __future__ import annotations

import uuid

import pytest
from sqlalchemy import text

from apps.api.core.database import get_session
from apps.api.graph.explain import shortest_path
from eval.scoring.graph import score_graph
from eval.types import EvalCitation, EvalMode, EvalOutputRow, GraphTrace, GraphTracePath


@pytest.mark.asyncio
async def test_semantic_edge_without_justification_is_untraversable(require_db):
    """
    Edge-justification removal test (semantic edge).

    Expected:
    - With >=1 edge_justification_span: path exists.
    - After deleting all edge_justification_span rows: path is not found when
      require_grounded_semantic=True.
    """
    async with get_session() as session:
        tx = await session.begin()
        try:
            # Deterministic pair not affected by framework-mined edges (currently P004 -> others).
            p1 = "P001"
            p2 = "P002"
            # Ensure no pre-existing SCHOLAR_LINK edges between them within this tx.
            await session.execute(
                text(
                    """
                    DELETE FROM edge_justification_span
                    WHERE edge_id IN (
                      SELECT id FROM edge
                      WHERE rel_type='SCHOLAR_LINK'
                        AND (
                          (from_type='pillar' AND from_id=:a AND to_type='pillar' AND to_id=:b)
                          OR
                          (from_type='pillar' AND from_id=:b AND to_type='pillar' AND to_id=:a)
                        )
                    )
                    """
                ),
                {"a": p1, "b": p2},
            )
            await session.execute(
                text(
                    """
                    DELETE FROM edge
                    WHERE rel_type='SCHOLAR_LINK'
                      AND (
                        (from_type='pillar' AND from_id=:a AND to_type='pillar' AND to_id=:b)
                        OR
                        (from_type='pillar' AND from_id=:b AND to_type='pillar' AND to_id=:a)
                      )
                    """
                ),
                {"a": p1, "b": p2},
            )

            chunk_row = (await session.execute(text("SELECT chunk_id, text_ar FROM chunk ORDER BY chunk_id LIMIT 1"))).fetchone()
            assert chunk_row is not None
            chunk_id = str(chunk_row.chunk_id)
            chunk_text = str(chunk_row.text_ar or "")
            assert chunk_text
            quote = chunk_text[: min(40, len(chunk_text))].strip() or chunk_text[:10]

            eid = uuid.uuid4()
            await session.execute(
                text(
                    """
                    INSERT INTO edge (
                      id, from_type, from_id, rel_type, relation_type, to_type, to_id,
                      created_method, created_by, justification, status
                    )
                    VALUES (
                      :id, 'pillar', :p1, 'SCHOLAR_LINK', 'COMPLEMENTS', 'pillar', :p2,
                      'human_approved', 'test', 'test_edge', 'approved'
                    )
                    """
                ),
                {"id": eid, "p1": p1, "p2": p2},
            )
            await session.execute(
                text(
                    """
                    INSERT INTO edge_justification_span (edge_id, chunk_id, span_start, span_end, quote)
                    VALUES (:eid, :cid, 0, :end, :q)
                    """
                ),
                {"eid": eid, "cid": chunk_id, "end": min(40, len(chunk_text)), "q": quote},
            )

            # With justification spans: should be traversable.
            res_ok = await shortest_path(
                session,
                start_type="pillar",
                start_id=p1,
                target_type="pillar",
                target_id=p2,
                max_depth=1,
                rel_types=["SCHOLAR_LINK"],
                require_grounded_semantic=True,
            )
            assert bool(res_ok.get("found")) is True

            # Remove justifications.
            await session.execute(text("DELETE FROM edge_justification_span WHERE edge_id=:eid"), {"eid": eid})

            # Without justification spans: must not be traversable.
            res_bad = await shortest_path(
                session,
                start_type="pillar",
                start_id=p1,
                target_type="pillar",
                target_id=p2,
                max_depth=1,
                rel_types=["SCHOLAR_LINK"],
                require_grounded_semantic=True,
            )
            assert bool(res_bad.get("found")) is False
        finally:
            await tx.rollback()


@pytest.mark.asyncio
async def test_citation_shuffle_breaks_semantic_edge_groundedness(require_db):
    """
    Citation shuffle test (end-to-end scorer).

    We create two SCHOLAR_LINK edges with different justification chunk_ids.
    With correct citations: explanation_grounded_rate == 1.0.
    After swapping citations between rows: explanation_grounded_rate == 0.0.
    """
    async with get_session() as session:
        tx = await session.begin()
        try:
            # Deterministic trio (framework mining may create many existing edges).
            # We delete any existing edges between these pairs inside the tx to keep the test stable.
            p1, p2, p3 = "P001", "P002", "P003"
            for to_id in [p2, p3]:
                await session.execute(
                    text(
                        """
                        DELETE FROM edge_justification_span
                        WHERE edge_id IN (
                          SELECT id FROM edge
                          WHERE rel_type='SCHOLAR_LINK'
                            AND from_type='pillar' AND from_id=:p1
                            AND to_type='pillar' AND to_id=:to_id
                        )
                        """
                    ),
                    {"p1": p1, "to_id": to_id},
                )
                await session.execute(
                    text(
                        """
                        DELETE FROM edge
                        WHERE rel_type='SCHOLAR_LINK'
                          AND from_type='pillar' AND from_id=:p1
                          AND to_type='pillar' AND to_id=:to_id
                        """
                    ),
                    {"p1": p1, "to_id": to_id},
                )

            chunks = (await session.execute(text("SELECT chunk_id, text_ar FROM chunk ORDER BY chunk_id LIMIT 2"))).fetchall()
            assert len(chunks) >= 2
            c1_id, c1_text = str(chunks[0].chunk_id), str(chunks[0].text_ar or "")
            c2_id, c2_text = str(chunks[1].chunk_id), str(chunks[1].text_ar or "")
            assert c1_text and c2_text
            q1 = (c1_text[: min(40, len(c1_text))].strip() or c1_text[:10])
            q2 = (c2_text[: min(40, len(c2_text))].strip() or c2_text[:10])

            e1_uuid = str(uuid.uuid4())
            e2_uuid = str(uuid.uuid4())

            # Edge IDs in scorer use the natural key (from_type:from_id::REL::to_type:to_id)
            e1_id = f"pillar:{p1}::SCHOLAR_LINK::pillar:{p2}"
            e2_id = f"pillar:{p1}::SCHOLAR_LINK::pillar:{p3}"

            for eid, to_id in [(e1_uuid, p2), (e2_uuid, p3)]:
                await session.execute(
                    text(
                        """
                        INSERT INTO edge (
                          id, from_type, from_id, rel_type, relation_type, to_type, to_id,
                          created_method, created_by, justification, status
                        )
                        VALUES (
                          :id, 'pillar', :p1, 'SCHOLAR_LINK', 'COMPLEMENTS', 'pillar', :to_id,
                          'human_approved', 'test', 'test_edge', 'approved'
                        )
                        """
                    ),
                    {"id": eid, "p1": p1, "to_id": to_id},
                )

            await session.execute(
                text(
                    """
                    INSERT INTO edge_justification_span (edge_id, chunk_id, span_start, span_end, quote)
                    VALUES (:eid, :cid, 0, :end, :q)
                    """
                ),
                {"eid": e1_uuid, "cid": c1_id, "end": min(40, len(c1_text)), "q": q1},
            )
            await session.execute(
                text(
                    """
                    INSERT INTO edge_justification_span (edge_id, chunk_id, span_start, span_end, quote)
                    VALUES (:eid, :cid, 0, :end, :q)
                    """
                ),
                {"eid": e2_uuid, "cid": c2_id, "end": min(40, len(c2_text)), "q": q2},
            )

            dmap = {
                "t1": {
                    "type": "cross_pillar",
                    "required_graph_paths": [
                        {"rel_type": "SCHOLAR_LINK", "relation_type": "COMPLEMENTS", "edges": [e1_id], "justification": "test", "nodes": [f"pillar:{p1}", f"pillar:{p2}"]},
                    ],
                },
                "t2": {
                    "type": "cross_pillar",
                    "required_graph_paths": [
                        {"rel_type": "SCHOLAR_LINK", "relation_type": "COMPLEMENTS", "edges": [e2_id], "justification": "test", "nodes": [f"pillar:{p1}", f"pillar:{p3}"]},
                    ],
                },
            }

            row1 = EvalOutputRow(
                id="t1",
                mode=EvalMode.FULL_SYSTEM,
                question="x",
                answer_ar="تعريف المفهوم داخل الإطار\n- x",
                abstained=False,
                citations=[EvalCitation(source_id=c1_id, span_start=0, span_end=10, quote=q1)],
                graph_trace=GraphTrace(nodes=[f"pillar:{p1}", f"pillar:{p2}"], edges=[e1_id], paths=[GraphTracePath(nodes=[f"pillar:{p1}", f"pillar:{p2}"], edges=[e1_id], confidence=1.0)]),
            )
            row2 = EvalOutputRow(
                id="t2",
                mode=EvalMode.FULL_SYSTEM,
                question="y",
                answer_ar="تعريف المفهوم داخل الإطار\n- y",
                abstained=False,
                citations=[EvalCitation(source_id=c2_id, span_start=0, span_end=10, quote=q2)],
                graph_trace=GraphTrace(nodes=[f"pillar:{p1}", f"pillar:{p3}"], edges=[e2_id], paths=[GraphTracePath(nodes=[f"pillar:{p1}", f"pillar:{p3}"], edges=[e2_id], confidence=1.0)]),
            )

            m_ok = await score_graph(session=session, outputs=[row1, row2], dataset_by_id=dmap)
            assert m_ok.explanation_grounded_rate == 1.0

            # Shuffle citations between rows (swap).
            row1_bad = row1.model_copy(update={"citations": row2.citations})
            row2_bad = row2.model_copy(update={"citations": row1.citations})

            m_bad = await score_graph(session=session, outputs=[row1_bad, row2_bad], dataset_by_id=dmap)
            assert m_bad.explanation_grounded_rate == 0.0
        finally:
            await tx.rollback()

