"""
Evidence parsing integration test using the real framework DOCX.
"""

import json
from pathlib import Path

from apps.api.ingest.pipeline_framework import ingest_framework_docx


def test_repo_docx_contains_quran_and_hadith_patterns(tmp_path: Path):
    docx = Path("docs/source/framework_2025-10_v1.docx")
    canonical_out = tmp_path / "framework_2025-10_v1.canonical.json"
    chunks_out = tmp_path / "framework_2025-10_v1.chunks.jsonl"

    ingest_framework_docx(docx, canonical_out, chunks_out)

    data = json.loads(canonical_out.read_text(encoding="utf-8"))

    q = 0
    h = 0
    for pillar in data.get("pillars", []):
        for cv in pillar.get("core_values", []):
            for e in cv.get("evidence", []) or []:
                if e.get("evidence_type") == "quran":
                    q += 1
                if e.get("evidence_type") == "hadith":
                    h += 1
            for sv in cv.get("sub_values", []):
                for e in sv.get("evidence", []) or []:
                    if e.get("evidence_type") == "quran":
                        q += 1
                    if e.get("evidence_type") == "hadith":
                        h += 1

    assert q >= 1
    assert h >= 1


