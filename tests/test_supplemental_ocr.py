from __future__ import annotations

import json
from pathlib import Path

from apps.api.ingest.docx_reader import ParsedDocument, ParsedParagraph
from apps.api.ingest.supplemental_ocr import (
    save_supplemental_ocr_text,
    load_supplemental_ocr_paragraphs,
)


def test_save_and_load_supplemental_ocr_paragraphs(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("SUPPLEMENTAL_OCR_DIR", str(tmp_path / "supp"))

    source_hash = "a" * 64
    img_sha = "b" * 64
    out = save_supplemental_ocr_text(
        source_file_hash=source_hash,
        image_sha256=img_sha,
        filename="page1.png",
        lines=["سطر 1", "", "سطر 2"],
    )
    assert out.exists()
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["source_file_hash"] == source_hash
    assert payload["image_sha256"] == img_sha
    assert payload["lines"] == ["سطر 1", "سطر 2"]

    doc = ParsedDocument(
        doc_name="framework.docx",
        doc_hash=source_hash,
        paragraphs=[ParsedParagraph(para_index=0, text="base")],
        total_paragraphs=1,
    )
    paras = load_supplemental_ocr_paragraphs(doc)
    assert len(paras) == 2
    assert paras[0].text == "سطر 1"
    assert paras[1].text == "سطر 2"
    assert paras[0].source_anchor.startswith("userimg_")


