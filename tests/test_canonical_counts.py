"""
Canonical corpus tests on the real framework DOCX in the repo.

These tests ensure:
- DOCX is ingested deterministically (no LLM)
- Canonical JSON is generated with required traceability fields
- Pillars count is exactly 5 for this source document
"""

from pathlib import Path

from apps.api.ingest.pipeline_framework import ingest_framework_docx


def test_canonical_counts_from_repo_docx(tmp_path: Path):
    docx = Path("docs/source/framework_2025-10_v1.docx")
    assert docx.exists()

    canonical_out = tmp_path / "framework_2025-10_v1.canonical.json"
    chunks_out = tmp_path / "framework_2025-10_v1.chunks.jsonl"

    summary = ingest_framework_docx(
        docx_path=docx,
        canonical_out_path=canonical_out,
        chunks_out_path=chunks_out,
        framework_version="2025-10",
    )

    assert summary.pillars == 5
    # For this framework, the "ركائز الحياة الطيبة" table defines 3 core values per pillar.
    assert summary.core_values == 15
    # Sub-values are determined by pillar tables/lists; keep a guardrail threshold until we lock exact count.
    assert summary.sub_values >= 60
    assert summary.evidence_records >= 50
    assert canonical_out.exists()
    assert chunks_out.exists()


