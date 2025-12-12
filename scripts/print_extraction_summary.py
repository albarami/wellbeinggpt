"""
Print extracted pillar/core/sub tree to help validate against the DOCX.

Usage:
  python scripts/print_extraction_summary.py
"""

from __future__ import annotations

from apps.api.ingest.docx_reader import DocxReader
from apps.api.ingest.rule_extractor import RuleExtractor


def main() -> None:
    doc = DocxReader().read("docs/source/framework_2025-10_v1.docx")
    ex = RuleExtractor(framework_version="2025-10").extract(doc)
    print("pillars:", len(ex.pillars), "core:", ex.total_core_values, "sub:", ex.total_sub_values, "evidence_blocks:", ex.total_evidence)
    for p in ex.pillars:
        print(f"\nPILLAR: {p.name_ar} ({p.id})")
        for cv in p.core_values:
            print(f"  CV: {cv.name_ar} ({cv.id}) sub_values={len(cv.sub_values)}")
            for sv in cv.sub_values:
                print(f"    - SV: {sv.name_ar} ({sv.id})")


if __name__ == "__main__":
    main()


