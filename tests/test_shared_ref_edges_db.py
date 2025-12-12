import os
import uuid
from urllib.parse import urlparse

import pytest


@pytest.mark.asyncio
async def test_build_edges_creates_shares_ref_edges():
    """
    Proves the "network" knows shared verses/hadith across entities by building SHARES_REF edges
    from evidence.ref_norm.
    """
    if not os.getenv("DATABASE_URL") or os.getenv("RUN_DB_TESTS") != "1":
        pytest.skip("Requires DATABASE_URL and RUN_DB_TESTS=1")

    import psycopg2
    from pathlib import Path
    from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
    from sqlalchemy import text

    from apps.api.ingest.loader import (
        create_source_document,
        create_ingestion_run,
        load_pillar,
        load_core_value,
        load_evidence,
        build_edges_for_source,
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
        # Apply schema
        schema_sql = Path("db/schema.sql").read_text(encoding="utf-8")
        conn2 = psycopg2.connect(
            f"postgresql://{u.username}:{u.password}@{u.hostname}:{u.port or 5432}/{dbname}"
        )
        conn2.autocommit = True
        cur2 = conn2.cursor()
        cur2.execute(schema_sql)
        conn2.close()

        async with Session() as session:
            source_doc_id = await create_source_document(
                session=session,
                file_name="test.docx",
                file_hash="abc123",
                framework_version="2025-10",
            )
            run_id = await create_ingestion_run(session=session, source_doc_id=source_doc_id)

            await load_pillar(
                session=session,
                pillar_data={
                    "id": "P001",
                    "name_ar": "ركيزة 1",
                    "name_en": None,
                    "description_ar": None,
                    "source_anchor": {"anchor_type": "para", "anchor_id": "para_0"},
                },
                source_doc_id=source_doc_id,
                run_id=run_id,
            )
            await load_pillar(
                session=session,
                pillar_data={
                    "id": "P002",
                    "name_ar": "ركيزة 2",
                    "name_en": None,
                    "description_ar": None,
                    "source_anchor": {"anchor_type": "para", "anchor_id": "para_1"},
                },
                source_doc_id=source_doc_id,
                run_id=run_id,
            )

            await load_core_value(
                session=session,
                cv_data={
                    "id": "CV001",
                    "name_ar": "قيمة 1",
                    "name_en": None,
                    "source_anchor": {"anchor_type": "para", "anchor_id": "para_10"},
                    "sub_values": [],
                    "evidence": [],
                },
                pillar_id="P001",
                source_doc_id=source_doc_id,
                run_id=run_id,
            )
            await load_core_value(
                session=session,
                cv_data={
                    "id": "CV002",
                    "name_ar": "قيمة 2",
                    "name_en": None,
                    "source_anchor": {"anchor_type": "para", "anchor_id": "para_20"},
                    "sub_values": [],
                    "evidence": [],
                },
                pillar_id="P002",
                source_doc_id=source_doc_id,
                run_id=run_id,
            )

            # Same normalized ref under two different entities
            await load_evidence(
                session=session,
                entity_type="core_value",
                entity_id="CV001",
                evidence_type="quran",
                ref_raw="البقرة: 21",
                ref_norm="quran:2:21",
                text_ar="نص",
                source_doc_id=source_doc_id,
                source_anchor={"anchor_type": "para", "anchor_id": "para_100"},
                run_id=run_id,
                parse_status="success",
                surah_number=2,
                ayah_number=21,
            )
            await load_evidence(
                session=session,
                entity_type="core_value",
                entity_id="CV002",
                evidence_type="quran",
                ref_raw="البقرة: 21",
                ref_norm="quran:2:21",
                text_ar="نص",
                source_doc_id=source_doc_id,
                source_anchor={"anchor_type": "para", "anchor_id": "para_101"},
                run_id=run_id,
                parse_status="success",
                surah_number=2,
                ayah_number=21,
            )

            await build_edges_for_source(session=session, source_doc_id=source_doc_id)
            await session.commit()

            rows = (
                await session.execute(
                    text(
                        """
                        SELECT from_type, from_id, rel_type, to_type, to_id, justification
                        FROM edge
                        WHERE rel_type='SHARES_REF'
                        """
                    )
                )
            ).fetchall()
            assert len(rows) >= 1
            assert rows[0].rel_type == "SHARES_REF"
            assert "quran:quran:2:21" in (rows[0].justification or "")
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


