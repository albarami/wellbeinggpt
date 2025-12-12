"""
DOCX Image Extraction Utilities (ingestion-only).

Extracts embedded images from a DOCX file (zip container) so that the ingestion
pipeline can OCR image-based pages and then load results into:
- canonical JSON
- Postgres (entities/evidence/chunks)
- vector RAG (embeddings/index)
- graph edges

Important:
- We do NOT persist images in the repo.
- We only compute hashes and pass bytes to OCR during ingestion.
"""

from __future__ import annotations

import hashlib
import io
import zipfile
from dataclasses import dataclass


@dataclass(frozen=True)
class DocxImage:
    """An embedded image extracted from the DOCX container."""

    filename: str
    sha256: str
    content_type: str
    data: bytes


def extract_images_from_docx_bytes(docx_bytes: bytes) -> list[DocxImage]:
    """
    Extract embedded images from a DOCX (zip) bytes payload.

    Args:
        docx_bytes: Raw bytes of the .docx file.

    Returns:
        List of DocxImage items sorted by filename (stable order).
    """
    images: list[DocxImage] = []
    with zipfile.ZipFile(io.BytesIO(docx_bytes), "r") as zf:
        media_paths = [
            n
            for n in zf.namelist()
            if n.startswith("word/media/") and not n.endswith("/")
        ]
        for name in sorted(media_paths):
            data = zf.read(name)
            sha256 = hashlib.sha256(data).hexdigest()
            lower = name.lower()
            if lower.endswith(".png"):
                ctype = "image/png"
            elif lower.endswith(".jpg") or lower.endswith(".jpeg"):
                ctype = "image/jpeg"
            elif lower.endswith(".gif"):
                ctype = "image/gif"
            elif lower.endswith(".webp"):
                ctype = "image/webp"
            else:
                ctype = "application/octet-stream"
            images.append(
                DocxImage(
                    filename=name.split("/")[-1],
                    sha256=sha256,
                    content_type=ctype,
                    data=data,
                )
            )
    return images


