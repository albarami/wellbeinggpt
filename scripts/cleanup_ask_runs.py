"""Cleanup old ask runs (retention hygiene).

Reason:
- UI observability storage is append-only and must be bounded.
- This script deletes old rows deterministically based on configured retention.

Environment:
- ASK_RUN_RETENTION_DAYS (default: 30)
- ASK_RUN_MAX_ROWS (optional; if set, keeps only the newest N rows)
"""

from __future__ import annotations

import asyncio
import os
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

    days = int(os.getenv("ASK_RUN_RETENTION_DAYS", "30"))
    max_rows_raw = os.getenv("ASK_RUN_MAX_ROWS", "").strip()
    max_rows = int(max_rows_raw) if max_rows_raw else None

    async with get_session() as s:
        # Retention by age.
        if days > 0:
            await s.execute(
                text(
                    """
                    DELETE FROM ask_run
                    WHERE created_at < (NOW() - (:days || ' days')::interval)
                    """
                ),
                {"days": days},
            )

        # Optional bounding by row count (keep newest N).
        if max_rows is not None and max_rows > 0:
            await s.execute(
                text(
                    """
                    DELETE FROM ask_run
                    WHERE request_id IN (
                      SELECT request_id
                      FROM ask_run
                      ORDER BY created_at DESC
                      OFFSET :max_rows
                    )
                    """
                ),
                {"max_rows": max_rows},
            )


if __name__ == "__main__":
    asyncio.run(main())

