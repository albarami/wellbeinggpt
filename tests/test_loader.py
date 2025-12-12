"""
Tests for database loader module.

Note: These tests require a database connection to run.
Most tests are marked as skip for CI without DB.
"""

import pytest
import os
import uuid
from urllib.parse import urlparse
from unittest.mock import AsyncMock, MagicMock, patch


class TestCanonicalJsonExport:
    """Tests for canonical JSON export functionality."""

    def test_extraction_to_canonical_json_structure(self):
        """Test canonical JSON has correct structure."""
        from apps.api.ingest.canonical_json import extraction_to_canonical_json
        from apps.api.ingest.rule_extractor import (
            ExtractionResult,
            ExtractedPillar,
        )

        result = ExtractionResult(
            source_doc_id="DOC_test",
            source_file_hash="abc123",
            source_doc="docs/source/framework_2025-10_v1.docx",
            framework_version="2025-10",
            pillars=[
                ExtractedPillar(
                    id="P001",
                    name_ar="الحياة الروحية",
                    source_doc="docs/source/framework_2025-10_v1.docx",
                    source_hash="abc123",
                    source_anchor="para_0",
                    raw_text="الحياة الروحية",
                    para_index=0,
                )
            ],
        )

        canonical = extraction_to_canonical_json(result)

        assert "meta" in canonical
        assert "pillars" in canonical
        assert canonical["meta"]["source_doc_id"] == "DOC_test"
        assert canonical["meta"]["framework_version"] == "2025-10"
        assert len(canonical["pillars"]) == 1

    def test_save_and_load_canonical_json(self, tmp_path):
        """Test saving and loading canonical JSON."""
        from apps.api.ingest.canonical_json import (
            save_canonical_json,
            load_canonical_json,
        )

        data = {
            "meta": {"version": "1.0"},
            "pillars": [{"id": "P001", "name_ar": "اختبار"}],
        }

        output_path = tmp_path / "test.json"
        save_canonical_json(data, output_path)

        loaded = load_canonical_json(output_path)

        assert loaded["meta"]["version"] == "1.0"
        assert loaded["pillars"][0]["name_ar"] == "اختبار"


class TestLoaderFunctions:
    """Tests for loader functions (mock-based)."""

    @pytest.mark.asyncio
    async def test_load_canonical_json_returns_summary(self):
        """Test that loader returns summary statistics."""
        from apps.api.ingest.loader import load_canonical_json_to_db

        # Create mock session
        mock_session = AsyncMock()
        mock_session.execute = AsyncMock()

        canonical = {
            "meta": {
                "source_file_hash": "abc123",
                "framework_version": "2025-10",
            },
            "pillars": [
                {
                    "id": "P001",
                    "name_ar": "ركيزة",
                    "source_anchor": {},
                    "core_values": [
                        {
                            "id": "CV001",
                            "name_ar": "قيمة",
                            "source_anchor": {},
                            "sub_values": [
                                {
                                    "id": "SV001",
                                    "name_ar": "قيمة فرعية",
                                    "source_anchor": {},
                                }
                            ],
                        }
                    ],
                }
            ],
        }

        result = await load_canonical_json_to_db(
            mock_session, canonical, "test.docx"
        )

        assert "source_doc_id" in result
        assert "run_id" in result
        assert result["pillars"] == 1
        assert result["core_values"] == 1
        assert result["sub_values"] == 1


class TestLoaderIntegration:
    """Integration tests for database loader."""

    @pytest.mark.asyncio
    async def test_load_pillar_to_db(self):
        """Test loading a pillar into the database."""
        if not os.getenv("DATABASE_URL") or os.getenv("RUN_DB_TESTS") != "1":
            pytest.skip("Requires DATABASE_URL and RUN_DB_TESTS=1")

        from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
        from sqlalchemy import text
        import psycopg2
        from pathlib import Path

        from apps.api.ingest.loader import create_source_document, create_ingestion_run, load_pillar

        db_url = os.environ["DATABASE_URL"]
        if db_url.startswith("postgresql://"):
            db_url_async = db_url.replace("postgresql://", "postgresql+asyncpg://", 1)
        else:
            db_url_async = db_url

        # Create an isolated test database
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
                        "name_ar": "الحياة الروحية",
                        "name_en": None,
                        "description_ar": None,
                        "source_anchor": {"anchor_type": "para", "anchor_id": "para_0"},
                    },
                    source_doc_id=source_doc_id,
                    run_id=run_id,
                )
                await session.commit()

                row = (
                    await session.execute(
                        text("SELECT id, name_ar, source_doc_id FROM pillar WHERE id='P001'")
                    )
                ).fetchone()
                assert row is not None
                assert row.id == "P001"
                assert row.name_ar == "الحياة الروحية"
                assert str(row.source_doc_id) == str(source_doc_id)
        finally:
            await engine.dispose()
            # Drop test database
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

    @pytest.mark.asyncio
    async def test_load_core_value_to_db(self):
        """Test loading a core value into the database."""
        if not os.getenv("DATABASE_URL") or os.getenv("RUN_DB_TESTS") != "1":
            pytest.skip("Requires DATABASE_URL and RUN_DB_TESTS=1")

        from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
        from sqlalchemy import text
        import psycopg2
        from pathlib import Path

        from apps.api.ingest.loader import (
            create_source_document,
            create_ingestion_run,
            load_pillar,
            load_core_value,
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
                        "name_ar": "الحياة الروحية",
                        "name_en": None,
                        "description_ar": None,
                        "source_anchor": {"anchor_type": "para", "anchor_id": "para_0"},
                    },
                    source_doc_id=source_doc_id,
                    run_id=run_id,
                )

                await load_core_value(
                    session=session,
                    cv_data={
                        "id": "CV001",
                        "name_ar": "الإخلاص",
                        "name_en": None,
                        "source_anchor": {"anchor_type": "para", "anchor_id": "para_10"},
                        "definition": {
                            "text_ar": "تعريف مختصر",
                            "source_anchor": {"anchor_type": "para", "anchor_id": "para_11"},
                        },
                        "sub_values": [],
                        "evidence": [],
                    },
                    pillar_id="P001",
                    source_doc_id=source_doc_id,
                    run_id=run_id,
                )
                await session.commit()

                row = (
                    await session.execute(
                        text("SELECT id, pillar_id, name_ar, definition_ar FROM core_value WHERE id='CV001'")
                    )
                ).fetchone()
                assert row is not None
                assert row.pillar_id == "P001"
                assert row.name_ar == "الإخلاص"
                assert row.definition_ar == "تعريف مختصر"

                tb = (
                    await session.execute(
                        text(
                            "SELECT entity_type, entity_id, block_type, text_ar FROM text_block WHERE entity_id='CV001'"
                        )
                    )
                ).fetchone()
                assert tb is not None
                assert tb.entity_type == "core_value"
                assert tb.block_type == "definition"
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

    @pytest.mark.asyncio
    async def test_create_ingestion_run(self):
        """Test creating an ingestion run record."""
        if not os.getenv("DATABASE_URL") or os.getenv("RUN_DB_TESTS") != "1":
            pytest.skip("Requires DATABASE_URL and RUN_DB_TESTS=1")

        from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
        from sqlalchemy import text
        import psycopg2
        from pathlib import Path

        from apps.api.ingest.loader import create_source_document, create_ingestion_run

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
                source_doc_id = await create_source_document(
                    session=session,
                    file_name="test.docx",
                    file_hash="abc123",
                    framework_version="2025-10",
                )
                run_id = await create_ingestion_run(session=session, source_doc_id=source_doc_id)
                await session.commit()

                row = (
                    await session.execute(
                        text("SELECT id, source_doc_id, status FROM ingestion_run WHERE id = :id"),
                        {"id": run_id},
                    )
                ).fetchone()
                assert row is not None
                assert str(row.source_doc_id) == str(source_doc_id)
                assert row.status == "in_progress"
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
            cur3.execute(f'DROP DATABASE IF EXISTS \"{dbname}\"')
            conn3.close()

