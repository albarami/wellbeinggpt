"""
Print a window of DOCX paragraphs around a keyword to understand formatting (style, anchors).

Usage:
  python scripts/inspect_docx_context.py الإيمان
"""

from __future__ import annotations

import sys
from apps.api.ingest.docx_reader import DocxReader


def main() -> None:
    arg = sys.argv[1] if len(sys.argv) > 1 else ""
    # Allow passing a paragraph index to avoid shell encoding issues with Arabic.
    keyword = arg or "الإيمان"
    doc = DocxReader().read("docs/source/framework_2025-10_v1.docx")

    hits: list[int] = []
    if keyword.isdigit():
        hits = [int(keyword)]
    else:
        for p in doc.paragraphs:
            t = (p.text or "").strip()
            if keyword in t and len(t) <= 80:
                hits.append(p.para_index)
                if len(hits) >= 5:
                    break

    if not hits:
        print("No hits for:", keyword)
        return

    start = max(0, hits[0] - 10)
    end = min(len(doc.paragraphs), hits[0] + 80)

    print("keyword:", keyword)
    print("first_hit_para_index:", hits[0])
    for p in doc.paragraphs[start:end]:
        t = (p.text or "").strip()
        if not t:
            continue
        print(f"{p.para_index}\t{p.source_anchor}\t{p.style}\t{t[:160]}")


if __name__ == "__main__":
    main()


