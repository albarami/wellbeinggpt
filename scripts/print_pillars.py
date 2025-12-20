"""Print pillar IDs + Arabic names from DB (UTF-8)."""

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
        rows = (await session.execute(text("SELECT id, name_ar FROM pillar ORDER BY id"))).fetchall()
        for r in rows:
            print(f"{r.id}\t{r.name_ar}")


def main() -> None:
    asyncio.run(_run())


if __name__ == "__main__":
    main()

