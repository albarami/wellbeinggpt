"""CLI: show grounded cross-pillar SCHOLAR_LINK edge counts.

Run:
  python -m scripts.bridge_diagnostics
  OR
  python scripts/bridge_diagnostics.py
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

# Add project root to path for direct script execution
sys.path.insert(0, str(Path(__file__).parent.parent))

from apps.api.core.database import get_session
from apps.api.graph.bridge_diagnostics import (
    count_grounded_cross_pillar_scholar_links,
    count_value_level_scholar_links,
    sample_grounded_cross_pillar_scholar_links,
)
from eval.datasets.source_loader import load_dotenv_if_present


async def _run() -> None:
    load_dotenv_if_present()
    async with get_session() as session:
        # Pillar-level edges
        n = await count_grounded_cross_pillar_scholar_links(session=session)
        print(f"\n=== Pillar-Level Edges ===")
        print(f"grounded_cross_pillar_scholar_links={n}")
        if n > 0:
            sample = await sample_grounded_cross_pillar_scholar_links(session=session, limit=10)
            for r in sample:
                print(r)

        # Value-level edges
        value_counts = await count_value_level_scholar_links(session=session)
        print(f"\n=== Value-Level Edges ===")
        print(f"total_value_edges={value_counts['total_value_edges']}")
        print(f"grounded_value_edges={value_counts['grounded_value_edges']}")
        print(f"cross_pillar_value_edges={value_counts['cross_pillar_value_edges']}")
        print(f"grounded_cross_pillar_value_edges={value_counts['grounded_cross_pillar_value_edges']}")


def main() -> None:
    asyncio.run(_run())


if __name__ == "__main__":
    main()

