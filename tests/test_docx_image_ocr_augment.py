"""
Tests for DOCX embedded-image OCR augmentation.

We do NOT call Azure in tests. Instead, we use a stub OCR client.
"""

import os
from pathlib import Path

import pytest

from apps.api.ingest.docx_reader import DocxReader
from apps.api.ingest.docx_images import extract_images_from_docx_bytes
from apps.api.ingest.ocr_augment import augment_document_with_image_ocr
from apps.api.llm.vision_ocr_azure import OcrResult


class _StubOcrClient:
    async def ocr_image(self, image_bytes: bytes) -> OcrResult:  # noqa: D401
        # Return multi-line text to verify splitting into pseudo-paragraphs.
        return OcrResult(
            image_sha256="stub",
            text_ar="الركيزة العاطفية\\nالقيم الجزئية\\nالانسجام\\nالتعريف الإجرائي: ...\\nالتأصيل: ...",
            error=None,
        )


def test_extract_images_from_repo_docx_is_stable():
    b = Path("docs/source/framework_2025-10_v1.docx").read_bytes()
    imgs = extract_images_from_docx_bytes(b)
    # Regression gate: this committed docx currently contains exactly 1 embedded image.
    assert len(imgs) == 1
    assert imgs[0].filename
    assert len(imgs[0].sha256) == 64
    assert len(imgs[0].data) > 1000


@pytest.mark.asyncio
async def test_ocr_augment_splits_into_paragraphs(monkeypatch):
    monkeypatch.setenv("INGEST_OCR_FROM_IMAGES", "required")
    monkeypatch.setenv("INGEST_OCR_MAX_IMAGES", "1")
    # Configure as "present" to satisfy required-mode gating (but we stub the client, no network).
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://example.invalid")
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "test")
    monkeypatch.setenv("AZURE_OPENAI_VISION_DEPLOYMENT_NAME", "test")

    b = Path("docs/source/framework_2025-10_v1.docx").read_bytes()
    reader = DocxReader()
    doc = reader.read_bytes(b, "framework_2025-10_v1.docx")
    base = doc.total_paragraphs

    new_doc, stats = await augment_document_with_image_ocr(doc, b, client=_StubOcrClient())
    assert stats.images_found >= 1
    assert stats.images_ocr_succeeded == 1
    assert new_doc.total_paragraphs > base

    # Ensure multiple lines became multiple paragraphs with docimg_ anchors.
    added = new_doc.paragraphs[base:]
    assert any(p.source_anchor.startswith("docimg_") for p in added)
    assert any("الانسجام" in (p.text or "") for p in added)


