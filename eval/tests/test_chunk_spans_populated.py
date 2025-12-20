"""DB integration: chunk_span table should be populated after eval bootstrap."""

import pytest

from sqlalchemy import text

from apps.api.core.database import get_session
from eval.db_bootstrap import ensure_db_populated, DbBootstrapConfig


@pytest.mark.asyncio
async def test_chunk_span_table_populated(require_db_strict):
    await ensure_db_populated(DbBootstrapConfig())

    async with get_session() as session:
        # There should be at least some spans.
        n = (
            await session.execute(text("SELECT COUNT(*) FROM chunk_span"))
        ).scalar_one()
        assert int(n) > 0

        # Spot-check span_index=0 exists for at least one chunk.
        n2 = (
            await session.execute(
                text("SELECT COUNT(*) FROM chunk_span WHERE span_index=0")
            )
        ).scalar_one()
        assert int(n2) > 0
