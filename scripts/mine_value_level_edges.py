"""CLI: Mine value-level semantic edges from framework corpus.

This script:
1. Scans all chunks for co-mentions of core_value/sub_value entities
2. Applies deterministic pattern matching for relation types
3. Inserts grounded SCHOLAR_LINK edges with justification spans
4. Reports mining statistics

Run:
  python -m scripts.mine_value_level_edges

Targets (per spec):
- >=300 value-level edges total
- >=80 cross-pillar value-level edges
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from apps.api.core.database import get_session
from apps.api.graph.value_edge_miner import (
    count_value_level_edges,
    mine_value_level_edges,
    upsert_value_edges,
)
from eval.datasets.source_loader import load_dotenv_if_present

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


async def _run(dry_run: bool = False, verbose: bool = False) -> int:
    """Run the value-level edge mining pipeline.

    Args:
        dry_run: If True, don't insert edges, just report what would be mined.
        verbose: If True, print sample edges.

    Returns:
        Exit code (0 = success, 1 = below targets)
    """
    load_dotenv_if_present()

    async with get_session() as session:
        # Check existing value-level edges
        existing = await count_value_level_edges(session)
        print(f"\n=== Existing Value-Level Edges ===")
        print(f"  total_value_edges={existing['total_value_edges']}")
        print(f"  grounded_value_edges={existing['grounded_value_edges']}")

        # Mine new edges
        print(f"\n=== Mining Value-Level Edges ===")
        edges, report = await mine_value_level_edges(session)

        print(f"\n{report.summary()}")

        if verbose and edges:
            print(f"\n=== Sample Edges (first 20) ===")
            for e in edges[:20]:
                cross = "CROSS" if e.from_pillar_id != e.to_pillar_id else "SAME"
                print(
                    f"  {e.from_type}:{e.from_id} --[{e.relation_type}]--> "
                    f"{e.to_type}:{e.to_id} ({cross}-pillar, spans={len(e.spans)})"
                )
                if e.spans:
                    quote = e.spans[0].quote[:80] + "..." if len(e.spans[0].quote) > 80 else e.spans[0].quote
                    print(f"    Quote: {quote}")

        if dry_run:
            print(f"\n=== DRY RUN - No edges inserted ===")
        else:
            # Insert edges
            print(f"\n=== Inserting Edges ===")
            counts = await upsert_value_edges(session=session, edges=edges)
            await session.commit()
            print(f"  inserted_edges={counts['inserted_edges']}")
            print(f"  inserted_spans={counts['inserted_spans']}")

            # Verify final counts
            final = await count_value_level_edges(session)
            print(f"\n=== Final Value-Level Edge Counts ===")
            print(f"  total_value_edges={final['total_value_edges']}")
            print(f"  grounded_value_edges={final['grounded_value_edges']}")

        # Check targets
        print(f"\n=== Target Check ===")
        total_ok = report.total_edges >= 300
        cross_ok = report.cross_pillar_edges >= 80
        print(f"  total_edges >= 300: {report.total_edges} {'OK' if total_ok else 'NEEDS_MORE'}")
        print(f"  cross_pillar >= 80: {report.cross_pillar_edges} {'OK' if cross_ok else 'NEEDS_MORE'}")

        if total_ok and cross_ok:
            print(f"\nAll targets met!")
            return 0
        else:
            print(f"\nTargets not fully met. See recommendations below.")
            if not total_ok:
                print(f"  - Need more value entity mentions in chunks")
                print(f"  - Consider adding value name aliases/variants")
            if not cross_ok:
                print(f"  - Need more cross-pillar value co-mentions")
                print(f"  - Check if chunks contain cross-pillar discussions")
            return 1


def main() -> None:
    parser = argparse.ArgumentParser(description="Mine value-level semantic edges")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Don't insert edges, just report what would be mined",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Print sample edges",
    )
    args = parser.parse_args()

    exit_code = asyncio.run(_run(dry_run=args.dry_run, verbose=args.verbose))
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
