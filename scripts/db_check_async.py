"""
Debug script: connect via SQLAlchemy asyncpg and print schema visibility.
"""

from __future__ import annotations

import os
import asyncio

from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine


async def main() -> None:
    url = os.getenv(
        "DATABASE_URL",
        "postgresql+asyncpg://postgres:1234@localhost:5432/wellbeing_db",
    )
    if url.startswith("postgresql://"):
        url = url.replace("postgresql://", "postgresql+asyncpg://", 1)
    engine = create_async_engine(url, echo=False)
    async with engine.connect() as conn:
        row = (
            await conn.execute(
                text("SELECT current_database(), current_user, inet_server_addr(), inet_server_port()")
            )
        ).fetchone()
        print("db/user:", row)
        sp = (await conn.execute(text("SHOW search_path"))).fetchone()
        print("search_path:", sp[0] if sp else None)
        tables = (
            await conn.execute(
                text(
                    """
                    SELECT table_schema, table_name
                    FROM information_schema.tables
                    WHERE table_type='BASE TABLE'
                      AND table_schema NOT IN ('pg_catalog','information_schema')
                    ORDER BY table_schema, table_name
                    """
                )
            )
        ).fetchall()
        print("tables:", len(tables))
        for t in tables:
            print("-", t[0], t[1])
        reg = (await conn.execute(text("SELECT to_regclass('source_document'), to_regclass('public.source_document')"))).fetchone()
        print("to_regclass:", reg)
    await engine.dispose()


if __name__ == "__main__":
    os.environ.setdefault(
        "DATABASE_URL",
        "postgresql+asyncpg://postgres:1234@127.0.0.1:5432/wellbeing_db",
    )
    asyncio.run(main())


