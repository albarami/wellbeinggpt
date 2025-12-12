"""
OCR Augmentation for Ingestion (DOCX-embedded images -> OCR paragraphs).

This module converts image-based pages into text *during ingestion* and feeds
the output back into the deterministic rule extractor.

Enterprise-grade constraints:
- Never used at runtime (/ask).
- Adds stable anchors that trace back to the DOCX container via image SHA256.
- Does not persist images; only text + hashes are stored downstream.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

from apps.api.ingest.docx_reader import ParsedDocument, ParsedParagraph
from apps.api.ingest.docx_images import extract_images_from_docx_bytes
from apps.api.llm.vision_ocr_azure import VisionOcrClient, VisionOcrConfig


@dataclass(frozen=True)
class OcrAugmentStats:
    images_found: int
    images_ocr_attempted: int
    images_ocr_succeeded: int
    images_ocr_failed: int


def _ocr_mode() -> str:
    """
    OCR mode:
    - "off": never OCR
    - "auto": OCR only if Vision OCR config is present
    - "required": OCR must run and must be configured
    """
    return os.getenv("INGEST_OCR_FROM_IMAGES", "auto").strip().lower()

def _ocr_max_images() -> int:
    """
    Hard safety limit for OCR images per ingestion.

    Reason: protects ingestion latency/cost in enterprise environments.
    """
    try:
        return int(os.getenv("INGEST_OCR_MAX_IMAGES", "5"))
    except Exception:
        return 5


def _split_ocr_text_to_paragraphs(text: str) -> list[str]:
    """
    Split OCR output into paragraph-like lines.

    Reason: the rule-based extractor expects headings/markers to appear at
    paragraph granularity. OCR often returns multi-line blocks.
    """
    if not text:
        return []
    out: list[str] = []
    for raw in text.splitlines():
        t = raw.strip()
        if t:
            out.append(t)
    return out


async def augment_document_with_image_ocr(
    doc: ParsedDocument,
    docx_bytes: bytes,
    *,
    client: Optional[VisionOcrClient] = None,
) -> tuple[ParsedDocument, OcrAugmentStats]:
    """
    Append OCR text as additional ParsedParagraphs at the end of the document.

    Args:
        doc: ParsedDocument from DocxReader (selectable text + tables).
        docx_bytes: Original docx bytes (to extract embedded images).
        client: Optional injected VisionOcrClient (useful for tests).

    Returns:
        (doc, stats) where doc has extra OCR paragraphs appended.
    """
    mode = _ocr_mode()
    images = extract_images_from_docx_bytes(docx_bytes)
    if not images:
        return doc, OcrAugmentStats(0, 0, 0, 0)

    cfg = VisionOcrConfig.from_env()
    if mode == "off":
        return doc, OcrAugmentStats(len(images), 0, 0, 0)

    if mode == "required" and not cfg.is_configured():
        raise RuntimeError(
            "INGEST_OCR_FROM_IMAGES=required but Vision OCR is not configured. "
            "Set AZURE_OPENAI_ENDPOINT/AZURE_OPENAI_API_KEY and AZURE_OPENAI_VISION_DEPLOYMENT_NAME."
        )

    if mode == "auto" and not cfg.is_configured():
        return doc, OcrAugmentStats(len(images), 0, 0, 0)

    ocr_client = client or VisionOcrClient(cfg)

    appended: list[ParsedParagraph] = []
    base_idx = len(doc.paragraphs)
    attempted = 0
    ok = 0
    failed = 0

    # OCR images. Note: stable order is by filename (from extractor).
    max_images = _ocr_max_images()
    for i, img in enumerate(images[:max_images]):
        attempted += 1
        res = await ocr_client.ocr_image(img.data)
        if res.error or not (res.text_ar or "").strip():
            failed += 1
            continue

        ok += 1
        # Split OCR into pseudo-paragraphs for the extractor.
        # Anchor traces to DOCX media via sha256 + filename + line number.
        for ln, line in enumerate(_split_ocr_text_to_paragraphs(res.text_ar.strip())):
            anchor = f"docimg_{i}_{img.filename}_{img.sha256[:12]}_ln{ln}"
            appended.append(
                ParsedParagraph(
                    para_index=base_idx + len(appended),
                    text=line,
                    style="OCR_IMAGE",
                    doc_name=doc.doc_name,
                    doc_hash=doc.doc_hash,
                    source_anchor=anchor,
                    runs=[],
                )
            )

    # Return a new ParsedDocument with augmented paragraphs.
    new_doc = ParsedDocument(
        doc_name=doc.doc_name,
        doc_hash=doc.doc_hash,
        paragraphs=list(doc.paragraphs) + appended,
        total_paragraphs=len(doc.paragraphs) + len(appended),
    )

    return new_doc, OcrAugmentStats(
        images_found=len(images),
        images_ocr_attempted=attempted,
        images_ocr_succeeded=ok,
        images_ocr_failed=failed,
    )


