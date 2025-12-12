"""
Tests for DOCX extraction.

Tests the docx_reader module for parsing Arabic .docx files
with stable anchors.
"""

import pytest
from unittest.mock import Mock, patch
import hashlib

from apps.api.ingest.docx_reader import (
    DocxReader,
    ParsedParagraph,
    ParsedDocument,
    create_source_anchor,
)


class TestParsedParagraph:
    """Tests for ParsedParagraph dataclass."""

    def test_text_hash_computed(self):
        """Test that text hash is computed automatically."""
        para = ParsedParagraph(
            para_index=0,
            text="اختبار النص العربي",
        )
        assert para.text_hash != ""
        assert len(para.text_hash) == 12  # SHA1 truncated to 12 chars

    def test_anchor_id_with_para_id(self):
        """Test anchor_id prefers w14:paraId when available."""
        para = ParsedParagraph(
            para_index=5,
            text="بعض النص",
            para_id="ABC123",
        )
        assert para.anchor_id == "pid_ABC123"

    def test_anchor_id_fallback(self):
        """Test anchor_id falls back to para_index + hash."""
        para = ParsedParagraph(
            para_index=5,
            text="بعض النص",
        )
        assert para.anchor_id.startswith("p5_")
        assert len(para.anchor_id) > 3

    def test_empty_text_hash(self):
        """Test empty text produces empty hash."""
        para = ParsedParagraph(
            para_index=0,
            text="",
        )
        assert para.text_hash == ""

    def test_normalize_for_hash_removes_diacritics(self):
        """Test normalization removes diacritics before hashing."""
        # Same word with and without diacritics (tashkeel)
        para1 = ParsedParagraph(para_index=0, text="الْإِيمَانُ")  # with diacritics
        para2 = ParsedParagraph(para_index=0, text="الإيمان")  # without diacritics
        # Both should normalize similarly (diacritics removed)
        assert para1.text_hash == para2.text_hash


class TestParsedDocument:
    """Tests for ParsedDocument dataclass."""

    def test_total_paragraphs_computed(self):
        """Test total_paragraphs is computed from list."""
        paras = [
            ParsedParagraph(para_index=i, text=f"Paragraph {i}")
            for i in range(5)
        ]
        doc = ParsedDocument(
            file_name="test.docx",
            file_hash="abc123",
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


class TestCreateSourceAnchor:
    """Tests for create_source_anchor function."""

    def test_anchor_structure(self):
        """Test anchor dictionary has correct structure."""
        para = ParsedParagraph(
            para_index=10,
            text="Test paragraph",
            para_id="XYZ789",
        )
        doc_hash = "a" * 64

        anchor = create_source_anchor(doc_hash, para)

        assert "source_doc_id" in anchor
        assert "anchor_type" in anchor
        assert "anchor_id" in anchor
        assert "anchor_range" in anchor

    def test_source_doc_id_format(self):
        """Test source_doc_id uses first 16 chars of hash."""
        para = ParsedParagraph(para_index=0, text="Test")
        doc_hash = "abcdef1234567890" + "0" * 48

        anchor = create_source_anchor(doc_hash, para)

        assert anchor["source_doc_id"] == "DOC_abcdef1234567890"

    def test_anchor_type_is_docx_para(self):
        """Test anchor_type is set to docx_para."""
        para = ParsedParagraph(para_index=0, text="Test")

        anchor = create_source_anchor("x" * 64, para)

        assert anchor["anchor_type"] == "docx_para"

    def test_anchor_range_can_be_set(self):
        """Test anchor_range can be provided."""
        para = ParsedParagraph(para_index=0, text="Test")

        anchor = create_source_anchor(
            "x" * 64, para, anchor_range="p0-p5"
        )

        assert anchor["anchor_range"] == "p0-p5"


class TestDocxReaderIntegration:
    """Integration tests that would require actual .docx files."""

    @pytest.mark.skip(reason="Requires sample .docx file")
    def test_parse_real_docx(self):
        """Test parsing a real Arabic .docx file."""
        # This test would use a fixture file
        pass

    @pytest.mark.skip(reason="Requires sample .docx file")
    def test_extract_paragraph_ids(self):
        """Test extraction of w14:paraId from document XML."""
        pass

