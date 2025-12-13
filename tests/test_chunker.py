"""
Tests for the chunker module.

Tests chunk generation from canonical JSON data.
"""

import pytest

from apps.api.ingest.chunker import (
    Chunker,
    Chunk,
    chunk_to_dict,
    chunk_to_evidence_packet,
)
from apps.api.core.schemas import EntityType, ChunkType


class TestChunker:
    """Tests for the Chunker class."""

    def test_chunk_empty_data(self):
        """Test chunking empty canonical data."""
        chunker = Chunker()
        chunks = chunker.chunk_canonical_json({})

        assert len(chunks) == 0

    def test_chunk_pillar_description(self):
        """Test that pillar descriptions are chunked."""
        canonical = {
            "meta": {"source_doc_id": "DOC_test"},
            "pillars": [{
                "id": "P001",
                "name_ar": "الحياة الروحية",
                "description_ar": "وصف الركيزة الروحية",
                "source_anchor": "para_0",
                "core_values": [],
            }]
        }

        chunker = Chunker()
        chunks = chunker.chunk_canonical_json(canonical)

        assert len(chunks) == 1
        assert chunks[0].entity_type == EntityType.PILLAR
        assert chunks[0].entity_id == "P001"
        assert chunks[0].chunk_type == ChunkType.DEFINITION

    def test_chunk_core_value_definition(self):
        """Test that core value definitions are chunked."""
        canonical = {
            "meta": {"source_doc_id": "DOC_test"},
            "pillars": [{
                "id": "P001",
                "name_ar": "الحياة الروحية",
                "core_values": [{
                    "id": "CV001",
                    "name_ar": "الإيمان",
                    "definition": {
                        "text_ar": "تعريف الإيمان هو التصديق والإقرار",
                        "source_anchor": "para_1",
                    },
                    "sub_values": [],
                }],
            }]
        }

        chunker = Chunker()
        chunks = chunker.chunk_canonical_json(canonical)

        assert len(chunks) == 1
        assert chunks[0].entity_type == EntityType.CORE_VALUE
        assert chunks[0].entity_id == "CV001"
        assert "تعريف الإيمان" in chunks[0].text_ar

    def test_chunk_sub_value_definition(self):
        """Test that sub-value definitions are chunked."""
        canonical = {
            "meta": {"source_doc_id": "DOC_test"},
            "pillars": [{
                "id": "P001",
                "name_ar": "الحياة الروحية",
                "core_values": [{
                    "id": "CV001",
                    "name_ar": "الإيمان",
                    "sub_values": [{
                        "id": "SV001",
                        "name_ar": "التوحيد",
                        "definition": {
                            "text_ar": "التوحيد هو إفراد الله بالعبادة",
                            "source_anchor": "para_2",
                        },
                    }],
                }],
            }]
        }

        chunker = Chunker()
        chunks = chunker.chunk_canonical_json(canonical)

        assert len(chunks) == 1
        assert chunks[0].entity_type == EntityType.SUB_VALUE
        assert chunks[0].entity_id == "SV001"

    def test_chunk_ids_are_unique(self):
        """Test that generated chunk IDs are unique."""
        canonical = {
            "meta": {"source_doc_id": "DOC_test"},
            "pillars": [{
                "id": "P001",
                "name_ar": "الحياة الروحية",
                "description_ar": "وصف أول",
                "core_values": [{
                    "id": "CV001",
                    "name_ar": "الإيمان",
                    "definition": {"text_ar": "تعريف"},
                    "sub_values": [{
                        "id": "SV001",
                        "name_ar": "التوحيد",
                        "definition": {"text_ar": "تعريف التوحيد"},
                    }],
                }],
            }]
        }

        chunker = Chunker()
        chunks = chunker.chunk_canonical_json(canonical)

        chunk_ids = [c.chunk_id for c in chunks]
        assert len(chunk_ids) == len(set(chunk_ids))  # All unique

    def test_evidence_chunk_ids_do_not_collide_on_same_text(self):
        """
        Evidence chunks must not collide when the same text+anchor carries different refs.

        This happens when an OCR evidence block is expanded into multiple refs.
        """
        canonical = {
            "meta": {"source_doc_id": "DOC_test"},
            "pillars": [{
                "id": "P001",
                "name_ar": "الحياة الروحية",
                "core_values": [{
                    "id": "CV001",
                    "name_ar": "الإيمان",
                    "sub_values": [{
                        "id": "SV001",
                        "name_ar": "التوحيد",
                        "evidence": [
                            {
                                "evidence_type": "quran",
                                "ref_raw": "[البقرة: 1]",
                                "text_ar": "نص واحد",
                                "source_anchor": "para_9",
                            },
                            {
                                "evidence_type": "quran",
                                "ref_raw": "[البقرة: 2]",
                                "text_ar": "نص واحد",
                                "source_anchor": "para_9",
                            },
                        ],
                    }],
                }],
            }],
        }

        chunker = Chunker()
        chunks = [c for c in chunker.chunk_canonical_json(canonical) if c.chunk_type == ChunkType.EVIDENCE]
        assert len(chunks) == 2
        assert len({c.chunk_id for c in chunks}) == 2

    def test_token_estimation(self):
        """Test that token count is estimated."""
        canonical = {
            "meta": {"source_doc_id": "DOC_test"},
            "pillars": [{
                "id": "P001",
                "name_ar": "ركيزة",
                "description_ar": "كلمة كلمة كلمة كلمة كلمة",  # 5 words
                "core_values": [],
            }]
        }

        chunker = Chunker()
        chunks = chunker.chunk_canonical_json(canonical)

        assert chunks[0].token_count_estimate > 0
        # 5 words * 1.5 = 7.5, rounded to 7
        assert chunks[0].token_count_estimate >= 5

    def test_long_text_is_split(self):
        """Test that long definition text is split into multiple chunks."""
        # Create very long definition (definitions are split, descriptions are not)
        long_text = " ".join(["كلمة"] * 500)  # 500 words

        canonical = {
            "meta": {"source_doc_id": "DOC_test"},
            "pillars": [{
                "id": "P001",
                "name_ar": "ركيزة",
                "core_values": [{
                    "id": "CV001",
                    "name_ar": "قيمة",
                    "definition": {"text_ar": long_text},  # Definition gets split
                    "sub_values": [],
                }],
            }]
        }

        chunker = Chunker(max_tokens=100)
        chunks = chunker.chunk_canonical_json(canonical)

        # Should split into multiple chunks
        assert len(chunks) > 1


class TestChunkConversions:
    """Tests for chunk conversion functions."""

    def test_chunk_to_dict(self):
        """Test converting chunk to dictionary."""
        chunk = Chunk(
            chunk_id="CH_000001",
            entity_type=EntityType.CORE_VALUE,
            entity_id="CV001",
            chunk_type=ChunkType.DEFINITION,
            text_ar="نص عربي",
            text_en=None,
            source_doc_id="DOC_test",
            source_anchor="p1_abc",
            token_count_estimate=5,
        )

        result = chunk_to_dict(chunk)

        assert result["chunk_id"] == "CH_000001"
        assert result["entity_type"] == "core_value"
        assert result["chunk_type"] == "definition"
        assert result["text_ar"] == "نص عربي"

    def test_chunk_to_evidence_packet(self):
        """Test converting chunk to evidence packet format."""
        chunk = Chunk(
            chunk_id="CH_000001",
            entity_type=EntityType.SUB_VALUE,
            entity_id="SV001",
            chunk_type=ChunkType.EVIDENCE,
            text_ar="﴿آية﴾",
            text_en=None,
            source_doc_id="DOC_test",
            source_anchor="p5_xyz",
            token_count_estimate=3,
            refs=[{"type": "quran", "ref": "البقرة:1"}],
        )

        packet = chunk_to_evidence_packet(chunk)

        # Should match Evidence Packet schema
        assert packet["chunk_id"] == "CH_000001"
        assert packet["entity_type"] == "sub_value"
        assert packet["entity_id"] == "SV001"
        assert packet["chunk_type"] == "evidence"
        assert packet["text_ar"] == "﴿آية﴾"
        assert packet["source_doc_id"] == "DOC_test"
        assert packet["source_anchor"] == "p5_xyz"
        assert len(packet["refs"]) == 1
        assert packet["refs"][0]["type"] == "quran"


class TestChunkerEdgeCases:
    """Edge case tests for chunker."""

    def test_empty_definition(self):
        """Test handling of empty definition."""
        canonical = {
            "meta": {"source_doc_id": "DOC_test"},
            "pillars": [{
                "id": "P001",
                "name_ar": "ركيزة",
                "core_values": [{
                    "id": "CV001",
                    "name_ar": "قيمة",
                    "definition": {"text_ar": ""},  # Empty
                    "sub_values": [],
                }],
            }]
        }

        chunker = Chunker()
        chunks = chunker.chunk_canonical_json(canonical)

        # Should not create chunk for empty text
        assert len(chunks) == 0

    def test_missing_definition(self):
        """Test handling of missing definition."""
        canonical = {
            "meta": {"source_doc_id": "DOC_test"},
            "pillars": [{
                "id": "P001",
                "name_ar": "ركيزة",
                "core_values": [{
                    "id": "CV001",
                    "name_ar": "قيمة",
                    "definition": None,
                    "sub_values": [],
                }],
            }]
        }

        chunker = Chunker()
        chunks = chunker.chunk_canonical_json(canonical)

        assert len(chunks) == 0

    def test_missing_source_anchor(self):
        """Test handling of missing source anchor."""
        canonical = {
            "meta": {"source_doc_id": "DOC_test"},
            "pillars": [{
                "id": "P001",
                "name_ar": "ركيزة",
                "description_ar": "وصف",
                "source_anchor": None,  # Missing
                "core_values": [],
            }]
        }

        chunker = Chunker()
        chunks = chunker.chunk_canonical_json(canonical)

        # Should still create chunk with "unknown" anchor
        assert len(chunks) == 1
        assert chunks[0].source_anchor == "unknown"

