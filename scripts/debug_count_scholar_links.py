"""Debug: count SCHOLAR_LINK edges touching a pillar."""

from __future__ import annotations

import asyncio
import sys

from sqlalchemy import text

from apps.api.core.database import get_session
from eval.datasets.source_loader import load_dotenv_if_present


async def _run() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    except Exception:
        pass
    load_dotenv_if_present()
    async with get_session() as session:
        c1 = (
            await session.execute(
                text(
                    "SELECT COUNT(*) AS c FROM edge WHERE rel_type='SCHOLAR_LINK' AND to_type='pillar' AND to_id='P001'"
                )
            )
        ).fetchone()
        c2 = (
            await session.execute(
                text(
                    "SELECT COUNT(*) AS c FROM edge WHERE rel_type='SCHOLAR_LINK' AND from_type='pillar' AND from_id='P004'"
                )
            )
        ).fetchone()
        print("to_P001:", int(getattr(c1, "c", 0) or 0))
        print("from_P004:", int(getattr(c2, "c", 0) or 0))


def main() -> None:
    asyncio.run(_run())


if __name__ == "__main__":
    main()

