"""CLI: show grounded cross-pillar SCHOLAR_LINK edge counts.

Run:
  python -m scripts.bridge_diagnostics
"""

from __future__ import annotations

import asyncio

from apps.api.core.database import get_session
from apps.api.graph.bridge_diagnostics import (
    count_grounded_cross_pillar_scholar_links,
    sample_grounded_cross_pillar_scholar_links,
)
from eval.datasets.source_loader import load_dotenv_if_present


async def _run() -> None:
    load_dotenv_if_present()
    async with get_session() as session:
        n = await count_grounded_cross_pillar_scholar_links(session=session)
        print(f"grounded_cross_pillar_scholar_links={n}")
        if n > 0:
            sample = await sample_grounded_cross_pillar_scholar_links(session=session, limit=10)
            for r in sample:
                print(r)


def main() -> None:
    asyncio.run(_run())


if __name__ == "__main__":
    main()

