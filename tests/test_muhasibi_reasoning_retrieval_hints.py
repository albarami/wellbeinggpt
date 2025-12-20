from __future__ import annotations

import pytest
from sqlalchemy import text


class _CountingRetriever:
    """
    Wrap a real HybridRetriever while counting retrieve calls.

    This is not mock data: the underlying retrieval is real and uses the DB.
    """

    def __init__(self, inner):
        self.inner = inner
        self._session = None
        self.calls: list[str] = []

    async def retrieve(self, session, inputs):
        self.calls.append(str(getattr(inputs, "query", "")))
        return await self.inner.retrieve(session, inputs)


@pytest.mark.asyncio
async def test_action_questions_trigger_extra_retrieve_calls(require_db):
    """
    For action/habit style questions, Muḥāsibī should attempt deterministic hint queries
    (in addition to the base query) to improve recall.
    """
    from apps.api.core.database import get_session
    from apps.api.retrieve.entity_resolver import EntityResolver
    from apps.api.retrieve.hybrid_retriever import HybridRetriever
    from apps.api.core.muhasibi_state_machine import StateContext, create_middleware, MuhasibiState

    async with get_session() as session:
        # Pick a sub_value name that has chunks (any).
        row = (
            await session.execute(
                text(
                    """
                    SELECT sv.id, sv.name_ar
                    FROM sub_value sv
                    JOIN chunk c ON c.entity_type='sub_value' AND c.entity_id=sv.id
                    GROUP BY sv.id, sv.name_ar
                    ORDER BY sv.id
                    LIMIT 1
                    """
                )
            )
        ).fetchone()
        if not row:
            pytest.skip("No sub_value with chunks found")

        # Load resolver from DB (real data).
        pillars = (await session.execute(text("SELECT id, name_ar FROM pillar"))).fetchall()
        core_values = (await session.execute(text("SELECT id, name_ar FROM core_value"))).fetchall()
        sub_values = (await session.execute(text("SELECT id, name_ar FROM sub_value"))).fetchall()

        resolver = EntityResolver()
        resolver.load_entities(
            pillars=[{"id": r.id, "name_ar": r.name_ar} for r in pillars],
            core_values=[{"id": r.id, "name_ar": r.name_ar} for r in core_values],
            sub_values=[{"id": r.id, "name_ar": r.name_ar} for r in sub_values],
        )

        base_retriever = HybridRetriever()
        counting = _CountingRetriever(base_retriever)
        counting._session = session

        middleware = create_middleware(entity_resolver=resolver, retriever=counting, llm_client=None, guardrails=None)

        # LISTEN to populate detected_entities
        ctx = StateContext(question=f"كيف أطبق {row.name_ar}؟")
        await middleware._execute_state(MuhasibiState.LISTEN, ctx)

        # RETRIEVE should now use hints because question is action-style and entities exist
        await middleware._execute_state(MuhasibiState.PATH, ctx)
        await middleware._execute_state(MuhasibiState.RETRIEVE, ctx)

        assert len(counting.calls) >= 2, f"Expected >=2 retrieve calls, got {counting.calls}"


@pytest.mark.asyncio
async def test_definition_questions_do_not_force_hint_queries(require_db):
    """
    For pure definition-style questions, we should not force extra hint queries.
    """
    from apps.api.core.database import get_session
    from apps.api.retrieve.entity_resolver import EntityResolver
    from apps.api.retrieve.hybrid_retriever import HybridRetriever
    from apps.api.core.muhasibi_state_machine import StateContext, create_middleware, MuhasibiState

    async with get_session() as session:
        row = (
            await session.execute(
                text(
                    """
                    SELECT sv.id, sv.name_ar
                    FROM sub_value sv
                    JOIN chunk c ON c.entity_type='sub_value' AND c.entity_id=sv.id
                    GROUP BY sv.id, sv.name_ar
                    ORDER BY sv.id
                    LIMIT 1
                    """
                )
            )
        ).fetchone()
        if not row:
            pytest.skip("No sub_value with chunks found")

        pillars = (await session.execute(text("SELECT id, name_ar FROM pillar"))).fetchall()
        core_values = (await session.execute(text("SELECT id, name_ar FROM core_value"))).fetchall()
        sub_values = (await session.execute(text("SELECT id, name_ar FROM sub_value"))).fetchall()

        resolver = EntityResolver()
        resolver.load_entities(
            pillars=[{"id": r.id, "name_ar": r.name_ar} for r in pillars],
            core_values=[{"id": r.id, "name_ar": r.name_ar} for r in core_values],
            sub_values=[{"id": r.id, "name_ar": r.name_ar} for r in sub_values],
        )

        base_retriever = HybridRetriever()
        counting = _CountingRetriever(base_retriever)
        counting._session = session
        middleware = create_middleware(entity_resolver=resolver, retriever=counting, llm_client=None, guardrails=None)

        ctx = StateContext(question=f"ما تعريف {row.name_ar}؟")
        await middleware._execute_state(MuhasibiState.LISTEN, ctx)
        await middleware._execute_state(MuhasibiState.PATH, ctx)
        await middleware._execute_state(MuhasibiState.RETRIEVE, ctx)

        assert len(counting.calls) == 1, f"Expected 1 retrieve call, got {counting.calls}"





