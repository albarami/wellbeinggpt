"""
Database schema bootstrapper.

Why this exists:
- The loader expects Postgres tables (source_document, pillar, core_value, ...).
- On Windows environments without `psql`, we still need a deterministic way to apply `db/schema.sql`.

Safety:
- Schema statements are written with IF NOT EXISTS / idempotent DDL where possible.
- This module executes statements sequentially inside a single transaction.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine


@dataclass(frozen=True)
class SchemaApplyResult:
    """Result of applying schema statements."""

    statements_total: int
    statements_executed: int
    statements_skipped: int


def _split_sql_statements(sql: str) -> List[str]:
    """
    Split a SQL script into statements.

    Notes:
    - This is a minimal splitter suited for `db/schema.sql` (no stored procedures / $$ blocks).
    - Removes `--` line comments and collapses blank lines.
    """
    cleaned_lines: list[str] = []
    for line in sql.splitlines():
        s = line.strip()
        if not s:
            continue
        if s.startswith("--"):
            continue
        # Strip inline comment portions: "..." -- comment
        if "--" in s:
            s = s.split("--", 1)[0].strip()
            if not s:
                continue
        cleaned_lines.append(s)

    joined = "\n".join(cleaned_lines)
    parts = [p.strip() for p in joined.split(";")]
    return [p for p in parts if p]


async def apply_schema(
    engine: AsyncEngine,
    schema_path: str | Path = Path("db/schema.sql"),
) -> SchemaApplyResult:
    """
    Apply the schema SQL to the database.

    Args:
        engine: Async SQLAlchemy engine.
        schema_path: Path to the schema SQL file.

    Returns:
        SchemaApplyResult: Counts of executed/skipped statements.
    """
    p = Path(schema_path)
    sql = p.read_text(encoding="utf-8")
    statements = _split_sql_statements(sql)
    executed = 0
    skipped = 0

    async with engine.begin() as conn:
        for stmt in statements:
            try:
                await conn.execute(text(stmt))
                executed += 1
            except Exception:
                # Best-effort idempotency: some DDL may fail if extension isn't permitted, etc.
                skipped += 1
                continue

    return SchemaApplyResult(
        statements_total=len(statements),
        statements_executed=executed,
        statements_skipped=skipped,
    )


def _db_url_from_env() -> str:
    """
    Get async DB URL from env, matching apps.api.core.database behavior.
    """
    url = os.getenv(
        "DATABASE_URL",
        "postgresql+asyncpg://wellbeing:wellbeing_dev_password@127.0.0.1:5432/wellbeing_db",
    )
    if url.startswith("postgresql://"):
        url = url.replace("postgresql://", "postgresql+asyncpg://")
    if os.name == "nt" and os.getenv("DB_HOST_PREFER_IPV4", "true").lower() == "true":
        url = url.replace("@localhost:", "@127.0.0.1:")
    return url


async def bootstrap_db(schema_path: str | Path = Path("db/schema.sql")) -> SchemaApplyResult:
    """
    Create an engine from env and apply schema.
    """
    engine = create_async_engine(_db_url_from_env(), pool_pre_ping=True)
    try:
        return await apply_schema(engine, schema_path=schema_path)
    finally:
        await engine.dispose()


