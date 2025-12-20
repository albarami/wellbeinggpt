"""Source inventory gate.

This module provides a deterministic allowlist of sources that scholar notes are
permitted to cite.

Reason:
- Prevent regressions where a notes pack references unknown sources.
- Keep "framework-only" policy enforceable at ingestion time.
"""

from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession


@dataclass(frozen=True)
class SourceInventory:
    """Deterministic inventory of allowed source IDs."""

    allowed_source_doc_ids: set[str]

    def assert_allowed(self, source_id: str) -> None:
        """Raise if source_id is not in the inventory."""

        if source_id not in self.allowed_source_doc_ids:
            raise ValueError(f"source_id not in inventory: {source_id}")


async def build_source_inventory(session: AsyncSession) -> SourceInventory:
    """Build the source inventory from the database.

    Args:
        session: DB session.

    Returns:
        SourceInventory.
    """

    rows = (
        await session.execute(
            text(
                """
                SELECT id
                FROM source_document
                ORDER BY id
                """
            )
        )
    ).fetchall()
    return SourceInventory(allowed_source_doc_ids={str(r.id) for r in rows if getattr(r, "id", None)})
