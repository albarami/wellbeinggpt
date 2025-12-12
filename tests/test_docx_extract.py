"""
Tests for DOCX extraction.

Tests the docx_reader module for parsing Arabic .docx files
with deterministic anchors (para_<index>) and SHA256 doc hash.
"""

import pytest
import hashlib

from apps.api.ingest.docx_reader import (
    DocxReader,
    ParsedParagraph,
    ParsedDocument,
)


class TestParsedParagraph:
    """Tests for ParsedParagraph dataclass."""

    def test_source_anchor_default(self):
        """Test that source_anchor defaults to para_<index>."""
        para = ParsedParagraph(
            para_index=0,
            text="اختبار النص العربي",
        )
        assert para.source_anchor == "para_0"

    def test_empty_text_anchor(self):
        """Empty text still gets a source_anchor."""
        para = ParsedParagraph(
            para_index=0,
            text="",
        )
        assert para.source_anchor == "para_0"


class TestParsedDocument:
    """Tests for ParsedDocument dataclass."""

    def test_total_paragraphs_computed(self):
        """Test total_paragraphs is computed from list."""
        paras = [
            ParsedParagraph(para_index=i, text=f"Paragraph {i}")
            for i in range(5)
        ]
        doc = ParsedDocument(
            doc_name="test.docx",
            doc_hash="abc123",
            paragraphs=paras,
        )
        assert doc.total_paragraphs == 5


class TestDocxReader:
    """Tests for DocxReader class."""

    def test_file_not_found(self):
        """Test FileNotFoundError for missing file."""
        reader = DocxReader()
        with pytest.raises(FileNotFoundError):
            reader.read("nonexistent.docx")

    def test_invalid_extension(self, tmp_path):
        """Test ValueError for non-.docx file."""
        txt_file = tmp_path / "test.txt"
        txt_file.write_text("Hello")

        reader = DocxReader()
        with pytest.raises(ValueError, match="Not a .docx file"):
            reader.read(txt_file)

    def test_compute_file_hash(self, tmp_path):
        """Test file hash computation is deterministic."""
        # Create a simple file
        test_file = tmp_path / "test.bin"
        content = b"Test content for hashing"
        test_file.write_bytes(content)

        reader = DocxReader()
        expected_hash = hashlib.sha256(content).hexdigest()
        actual_hash = reader._compute_file_hash(test_file)

        assert actual_hash == expected_hash


class TestDocxReaderIntegration:
    """Integration tests that would require actual .docx files."""

    def test_parse_real_docx(self):
        """Test parsing the real framework docx in repo."""
        reader = DocxReader()
        doc = reader.read("docs/source/framework_2025-10_v1.docx")

        # Stable count for this committed file (acts as a regression gate)
        assert doc.total_paragraphs == 1026
        assert len(doc.doc_hash) == 64
        assert doc.doc_name == "framework_2025-10_v1.docx"

        # Anchor scheme: para_<index>
        assert doc.paragraphs[0].source_anchor == "para_0"
        assert doc.paragraphs[-1].source_anchor == f"para_{doc.total_paragraphs-1}"

        # Each paragraph must carry doc metadata
        for p in doc.paragraphs[:20]:
            assert p.doc_name == doc.doc_name
            assert p.doc_hash == doc.doc_hash

