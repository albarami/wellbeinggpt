"""Debug: print SCHOLAR_LINK neighbors for a pillar, including justification spans."""

from __future__ import annotations

import asyncio
import sys

from apps.api.core.database import get_session
from apps.api.retrieve.graph_retriever import get_entity_neighbors
from eval.datasets.source_loader import load_dotenv_if_present


async def _run() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    except Exception:
        pass
    load_dotenv_if_present()
    async with get_session() as session:
        neigh_any = await get_entity_neighbors(
            session,
            "pillar",
            "P001",
            relationship_types=None,
            direction="both",
            status="approved",
        )
        neigh_scholar = await get_entity_neighbors(
            session,
            "pillar",
            "P001",
            relationship_types=["SCHOLAR_LINK"],
            direction="both",
            status="approved",
        )
        print("neighbors_any_rel_type:", len(neigh_any))
        print("neighbors_scholar_link:", len(neigh_scholar))
        for n in neigh_scholar[:10]:
            print(
                {
                    "edge_id": n.get("edge_id"),
                    "direction": n.get("direction"),
                    "neighbor_type": n.get("neighbor_type"),
                    "neighbor_id": n.get("neighbor_id"),
                    "relation_type": n.get("relation_type"),
                    "spans": len(n.get("justification_spans") or []),
                }
            )


def main() -> None:
    asyncio.run(_run())


if __name__ == "__main__":
    main()

