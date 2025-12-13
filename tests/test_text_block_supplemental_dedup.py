"""
DB integration test: supplemental_ocr text blocks must not overwrite each other.

Enterprise requirement:
- Each user-provided screenshot must remain auditable via its own `userimg_*` anchor.
- Therefore, `block_type="supplemental_ocr"` must allow multiple rows per entity.
"""

from __future__ import annotations

import os
import uuid
from pathlib import Path
from urllib.parse import urlparse

import pytest


@pytest.mark.asyncio
async def test_supplemental_ocr_text_blocks_allow_multiple_anchors() -> None:
    """
    Ensure `load_text_block(..., block_type='supplemental_ocr')` dedupes by anchor,
    not by (entity_type, entity_id, block_type) alone.
    """
    if not os.getenv("DATABASE_URL") or os.getenv("RUN_DB_TESTS") != "1":
        pytest.skip("Requires DATABASE_URL and RUN_DB_TESTS=1")

    import psycopg2
    from sqlalchemy import text
    from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

    from apps.api.ingest.loader import (
        create_ingestion_run,
        create_source_document,
        load_text_block,
    )

    db_url = os.environ["DATABASE_URL"]
    if db_url.startswith("postgresql://"):
        db_url_async = db_url.replace("postgresql://", "postgresql+asyncpg://", 1)
    else:
        db_url_async = db_url

    u = urlparse(db_url_async.replace("postgresql+asyncpg://", "postgresql://", 1))
    dbname = f"wellbeing_test_{uuid.uuid4().hex[:8]}"
    admin_dsn = f"postgresql://{u.username}:{u.password}@{u.hostname}:{u.port or 5432}/postgres"

    conn = psycopg2.connect(admin_dsn)
    conn.autocommit = True
    cur = conn.cursor()
    cur.execute(f'CREATE DATABASE "{dbname}"')
    conn.close()

    test_db_url = f"postgresql+asyncpg://{u.username}:{u.password}@{u.hostname}:{u.port or 5432}/{dbname}"
    engine = create_async_engine(test_db_url, echo=False)
    Session = async_sessionmaker(engine, expire_on_commit=False)

    try:
        schema_sql = Path("db/schema.sql").read_text(encoding="utf-8")
        conn2 = psycopg2.connect(
            f"postgresql://{u.username}:{u.password}@{u.hostname}:{u.port or 5432}/{dbname}"
        )
        conn2.autocommit = True
        cur2 = conn2.cursor()
        cur2.execute(schema_sql)
        conn2.close()

        async with Session() as session:
            sd = await create_source_document(
                session=session,
                file_name="test.docx",
                file_hash="abc123",
                framework_version="2025-10",
            )
            run_id = await create_ingestion_run(session=session, source_doc_id=sd)

            # Same entity, different anchors -> must create 2 rows (not overwrite).
            await load_text_block(
                session=session,
                entity_type="sub_value",
                entity_id="SV_TEST",
                block_type="supplemental_ocr",
                text_ar="أ",
                source_doc_id=sd,
                source_anchor={"source_anchor": "userimg_aaaaaaaaaaaa_ln0"},
                run_id=run_id,
            )
            await load_text_block(
                session=session,
                entity_type="sub_value",
                entity_id="SV_TEST",
                block_type="supplemental_ocr",
                text_ar="ب",
                source_doc_id=sd,
                source_anchor={"source_anchor": "userimg_bbbbbbbbbbbb_ln0"},
                run_id=run_id,
            )
            await session.commit()

            n = (
                await session.execute(
                    text(
                        """
                        SELECT COUNT(*)
                        FROM text_block
                        WHERE source_doc_id = :sd
                          AND entity_type = 'sub_value'
                          AND entity_id = 'SV_TEST'
                          AND block_type = 'supplemental_ocr'
                        """
                    ),
                    {"sd": sd},
                )
            ).scalar_one()
            assert n == 2

    finally:
        await engine.dispose()
        conn3 = psycopg2.connect(admin_dsn)
        conn3.autocommit = True
        cur3 = conn3.cursor()
        cur3.execute(
            """
            SELECT pg_terminate_backend(pid)
            FROM pg_stat_activity
            WHERE datname = %s AND pid <> pg_backend_pid()
            """,
            (dbname,),
        )
        cur3.execute(f'DROP DATABASE IF EXISTS "{dbname}"')
        conn3.close()


