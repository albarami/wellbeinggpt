"""
Supplemental OCR ingestion for user-provided screenshots (ingestion-only).

Why:
- Some sections may be missing or low-quality in DOCX-embedded images.
- Users can provide screenshots to speed up high-fidelity extraction.

How:
- OCR screenshots to text (no interpretation).
- Persist *only* OCR text + hashes under data/derived/ (gitignored).
- On subsequent DOCX ingestion, we append these OCR paragraphs to the ParsedDocument.

Traceability:
- Anchors include the screenshot sha256 + line index: `userimg_<sha12>_ln<k>`
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path

from apps.api.ingest.docx_reader import ParsedDocument, ParsedParagraph


def _base_dir() -> Path:
    return Path(os.getenv("SUPPLEMENTAL_OCR_DIR", "data/derived/supplemental_ocr"))


def _doc_dir(source_file_hash: str) -> Path:
    return _base_dir() / source_file_hash


@dataclass(frozen=True)
class SupplementalOcrWriteResult:
    source_file_hash: str
    images_received: int
    images_written: int


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def save_supplemental_ocr_text(
    *,
    source_file_hash: str,
    image_sha256: str,
    filename: str,
    lines: list[str],
) -> Path:
    """
    Save OCR lines for a single screenshot under the source DOCX hash.
    """
    d = _doc_dir(source_file_hash)
    d.mkdir(parents=True, exist_ok=True)
    out = d / f"{image_sha256[:16]}_{Path(filename).name}.json"
    payload = {
        "source_file_hash": source_file_hash,
        "image_sha256": image_sha256,
        "filename": filename,
        "lines": [ln for ln in lines if (ln or "").strip()],
    }
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return out


def load_supplemental_ocr_paragraphs(doc: ParsedDocument) -> list[ParsedParagraph]:
    """
    Load all supplemental OCR lines for a given DOCX hash as ParsedParagraphs.
    """
    d = _doc_dir(doc.doc_hash)
    if not d.exists():
        return []

    paras: list[ParsedParagraph] = []
    base_idx = doc.total_paragraphs
    for p in sorted(d.glob("*.json")):
        try:
            payload = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        img_sha = str(payload.get("image_sha256") or "")
        lines = payload.get("lines") or []
        if not img_sha or not isinstance(lines, list):
            continue
        for ln_idx, line in enumerate([str(x).strip() for x in lines if str(x).strip()]):
            anchor = f"userimg_{img_sha[:12]}_ln{ln_idx}"
            paras.append(
                ParsedParagraph(
                    para_index=base_idx + len(paras),
                    text=line,
                    style="OCR_USER_IMAGE",
                    doc_name=doc.doc_name,
                    doc_hash=doc.doc_hash,
                    source_anchor=anchor,
                    runs=[],
                )
            )
    return paras


