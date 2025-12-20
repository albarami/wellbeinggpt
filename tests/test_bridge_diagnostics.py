from __future__ import annotations

import pytest


@pytest.mark.asyncio
async def test_count_grounded_cross_pillar_edges_runs(require_db) -> None:
    """
    Smoke test: diagnostics query should run on a populated DB.
    """

    from apps.api.core.database import get_session
    from apps.api.graph.bridge_diagnostics import count_grounded_cross_pillar_scholar_links

    async with get_session() as session:
        n = await count_grounded_cross_pillar_scholar_links(session=session)
        assert isinstance(n, int)
        assert n >= 0

