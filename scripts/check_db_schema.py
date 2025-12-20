"""Check DB schema for scholar upgrade.

Prints whether expected columns/tables exist.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

from sqlalchemy import text

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval.datasets.source_loader import load_dotenv_if_present  # noqa: E402
from apps.api.core.database import get_session  # noqa: E402


async def main() -> None:
    load_dotenv_if_present()
    async with get_session() as s:
        cols = (
            await s.execute(
                text(
                    """
                    SELECT column_name
                    FROM information_schema.columns
                    WHERE table_schema='public'
                      AND table_name='edge'
                      AND column_name IN ('relation_type','strength_score')
                    ORDER BY column_name
                    """
                )
            )
        ).fetchall()
        print("edge columns:", [c[0] for c in cols])

        reg = (await s.execute(text("SELECT to_regclass('public.edge_justification_span')"))).fetchone()
        print("edge_justification_span exists:", bool(reg and reg[0]))


if __name__ == "__main__":
    asyncio.run(main())
