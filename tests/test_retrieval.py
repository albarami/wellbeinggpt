"""
Tests for retrieval layer.

Tests entity resolution, merge/rank, and evidence packet generation.
"""

import pytest

from apps.api.retrieve.entity_resolver import EntityResolver, ResolvedEntity
from apps.api.retrieve.merge_rank import (
    MergeRanker,
    MergeResult,
    ScoredPacket,
    create_evidence_bundle,
)
from apps.api.retrieve.normalize_ar import normalize_for_matching
from apps.api.core.schemas import EntityType, ChunkType


class TestEntityResolver:
    """Tests for EntityResolver."""

    def test_load_entities(self):
        """Test loading entities into resolver."""
        resolver = EntityResolver()
        resolver.load_entities(
            pillars=[{"id": "P001", "name_ar": "الحياة الروحية"}],
            core_values=[{"id": "CV001", "name_ar": "الإيمان"}],
            sub_values=[{"id": "SV001", "name_ar": "التوحيد"}],
        )

        assert resolver.get_entity_name("P001") == "الحياة الروحية"
        assert resolver.get_entity_name("CV001") == "الإيمان"
        assert resolver.get_entity_name("SV001") == "التوحيد"

    def test_resolve_exact_match(self):
        """Test resolving an exact entity match."""
        resolver = EntityResolver()
        resolver.load_entities(
            pillars=[{"id": "P001", "name_ar": "الحياة الروحية"}],
            core_values=[{"id": "CV001", "name_ar": "الإيمان"}],
            sub_values=[],
        )

        results = resolver.resolve("ما هو الإيمان؟")

        assert len(results) >= 1
        # Should find الإيمان
        assert any(r.entity_id == "CV001" for r in results)

    def test_resolve_normalized_match(self):
        """Test that normalization helps matching."""
        resolver = EntityResolver()
        resolver.load_entities(
            pillars=[],
            core_values=[{"id": "CV001", "name_ar": "الإيمان"}],
            sub_values=[],
        )

        # Query without hamza should still match
        results = resolver.resolve("ما هو الايمان؟")

        assert len(results) >= 1

    def test_resolve_multiple_entities(self):
        """Test resolving multiple entities from query."""
        resolver = EntityResolver()
        resolver.load_entities(
            pillars=[{"id": "P001", "name_ar": "الحياة الروحية"}],
            core_values=[{"id": "CV001", "name_ar": "الإيمان"}],
            sub_values=[{"id": "SV001", "name_ar": "التوحيد"}],
        )

        results = resolver.resolve("علاقة الإيمان والتوحيد في الحياة الروحية")

        # Should find multiple matches
        assert len(results) >= 2

    def test_resolve_no_match(self):
        """Test resolving when no entities match."""
        resolver = EntityResolver()
        resolver.load_entities(
            pillars=[{"id": "P001", "name_ar": "الحياة الروحية"}],
            core_values=[],
            sub_values=[],
        )

        results = resolver.resolve("سؤال عن موضوع مختلف تماما")

        # May find partial matches or none
        # The important thing is no errors
        assert isinstance(results, list)


class TestMergeRanker:
    """Tests for MergeRanker."""

    def test_merge_empty_results(self):
        """Test merging empty results."""
        ranker = MergeRanker()
        result = ranker.merge([], [], [])

        assert result.total_found == 0
        assert result.evidence_packets == []

    def test_merge_deduplicates(self):
        """Test that duplicate chunk IDs are deduplicated."""
        ranker = MergeRanker()

        packet = {
            "chunk_id": "CH_000001",
            "entity_type": "core_value",
            "entity_id": "CV001",
            "chunk_type": "definition",
            "text_ar": "تعريف",
            "source_doc_id": "DOC_test",
            "source_anchor": "p1",
            "refs": [],
        }

        # Same packet from multiple sources
        result = ranker.merge(
            sql_results=[packet],
            vector_results=[packet.copy()],
            graph_results=[],
        )

        # Should have only one packet
        assert len(result.evidence_packets) == 1
        assert result.evidence_packets[0]["chunk_id"] == "CH_000001"

    def test_merge_ranks_by_score(self):
        """Test that results are ranked by score."""
        ranker = MergeRanker()

        packet1 = {
            "chunk_id": "CH_000001",
            "entity_type": "core_value",
            "entity_id": "CV001",
            "chunk_type": "definition",
            "text_ar": "تعريف أول",
            "source_doc_id": "DOC_test",
            "source_anchor": "p1",
            "refs": [],
        }

        packet2 = {
            "chunk_id": "CH_000002",
            "entity_type": "sub_value",
            "entity_id": "SV001",
            "chunk_type": "definition",
            "text_ar": "تعريف ثاني",
            "source_doc_id": "DOC_test",
            "source_anchor": "p2",
            "refs": [{"type": "quran", "ref": "البقرة:1"}],
        }

        # SQL results should score higher
        result = ranker.merge(
            sql_results=[packet1],
            vector_results=[packet2],
            graph_results=[],
        )

        assert len(result.evidence_packets) == 2
        # SQL packet should be first (higher score)
        assert result.evidence_packets[0]["chunk_id"] == "CH_000001"

    def test_merge_tracks_sources(self):
        """Test that sources used are tracked."""
        ranker = MergeRanker()

        packet = {
            "chunk_id": "CH_000001",
            "entity_type": "core_value",
            "entity_id": "CV001",
            "chunk_type": "definition",
            "text_ar": "تعريف",
            "source_doc_id": "DOC_test",
            "source_anchor": "p1",
            "refs": [],
        }

        result = ranker.merge(
            sql_results=[packet],
            vector_results=[],
            graph_results=[],
        )

        assert "sql" in result.sources_used

    def test_merge_has_definition_flag(self):
        """Test that has_definition flag is set correctly."""
        ranker = MergeRanker()

        definition_packet = {
            "chunk_id": "CH_000001",
            "entity_type": "core_value",
            "entity_id": "CV001",
            "chunk_type": "definition",
            "text_ar": "تعريف",
            "source_doc_id": "DOC_test",
            "source_anchor": "p1",
            "refs": [],
        }

        evidence_packet = {
            "chunk_id": "CH_000002",
            "entity_type": "core_value",
            "entity_id": "CV001",
            "chunk_type": "evidence",
            "text_ar": "دليل",
            "source_doc_id": "DOC_test",
            "source_anchor": "p2",
            "refs": [],
        }

        # With definition
        result = ranker.merge([definition_packet], [], [])
        assert result.has_definition is True
        assert result.has_evidence is False

        # With evidence
        result = ranker.merge([evidence_packet], [], [])
        assert result.has_definition is False
        assert result.has_evidence is True


class TestCreateEvidenceBundle:
    """Tests for create_evidence_bundle function."""

    def test_formats_packets_correctly(self):
        """Test that packets are formatted to schema."""
        raw_packets = [
            {
                "chunk_id": "CH_000001",
                "entity_type": "core_value",
                "entity_id": "CV001",
                "chunk_type": "definition",
                "text_ar": "تعريف",
                "source_doc_id": "DOC_test",
                "source_anchor": "p1",
                "refs": [{"type": "quran", "ref": "البقرة:1"}],
                "extra_field": "should_be_preserved",  # Extra fields ok
            }
        ]

        bundle = create_evidence_bundle(raw_packets)

        assert len(bundle) == 1
        packet = bundle[0]

        # Check required fields
        assert packet["chunk_id"] == "CH_000001"
        assert packet["entity_type"] == "core_value"
        assert packet["entity_id"] == "CV001"
        assert packet["chunk_type"] == "definition"
        assert packet["text_ar"] == "تعريف"
        assert packet["source_doc_id"] == "DOC_test"
        assert packet["source_anchor"] == "p1"
        assert len(packet["refs"]) == 1

    def test_handles_missing_fields(self):
        """Test that missing fields get defaults."""
        raw_packets = [
            {"chunk_id": "CH_000001"}
        ]

        bundle = create_evidence_bundle(raw_packets)

        assert len(bundle) == 1
        packet = bundle[0]
        assert packet["chunk_id"] == "CH_000001"
        assert packet["text_ar"] == ""
        assert packet["refs"] == []

    def test_empty_input(self):
        """Test with empty input."""
        bundle = create_evidence_bundle([])
        assert bundle == []


class TestArabicNormalizationConsistency:
    """Tests for consistent Arabic normalization across retrieval."""

    def test_normalization_is_idempotent(self):
        """Test that normalizing twice gives same result."""
        text = "الإِيمَانُ"

        once = normalize_for_matching(text)
        twice = normalize_for_matching(once)

        assert once == twice

    def test_normalization_removes_diacritics(self):
        """Test that diacritics are removed."""
        text = "الْإِيمَانُ"
        normalized = normalize_for_matching(text)

        # Should not contain diacritics
        assert "ُ" not in normalized
        assert "ْ" not in normalized
        assert "ِ" not in normalized

    def test_normalization_handles_alef_variants(self):
        """Test that Alef variants are normalized."""
        variants = ["الإيمان", "الايمان", "الأيمان"]
        normalized = [normalize_for_matching(v) for v in variants]

        # All should normalize to same (or similar)
        # At minimum, first two should be same
        assert normalized[0] == normalized[1]

    def test_query_and_entity_use_same_normalization(self):
        """Test that queries and entities use consistent normalization."""
        entity_name = "الإيمان"
        query = "ما هو الايمان"

        # Both should normalize consistently
        norm_entity = normalize_for_matching(entity_name)
        norm_query = normalize_for_matching(query)

        # Entity should be findable in normalized query
        assert norm_entity in norm_query

