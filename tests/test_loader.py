"""
Tests for database loader module.

Note: These tests require a database connection to run.
Most tests are marked as skip for CI without DB.
"""

import pytest
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

    @pytest.mark.skip(reason="Requires database connection")
    @pytest.mark.asyncio
    async def test_load_pillar_to_db(self):
        """Test loading a pillar into the database."""
        pass

    @pytest.mark.skip(reason="Requires database connection")
    @pytest.mark.asyncio
    async def test_load_core_value_to_db(self):
        """Test loading a core value into the database."""
        pass

    @pytest.mark.skip(reason="Requires database connection")
    @pytest.mark.asyncio
    async def test_create_ingestion_run(self):
        """Test creating an ingestion run record."""
        pass

