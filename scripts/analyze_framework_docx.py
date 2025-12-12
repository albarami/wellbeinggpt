"""
DOCX structure + content analyzer for the wellbeing framework.

Goal: "Read the document end-to-end" deterministically and produce a debug report
showing pillars -> core values -> sub-values plus evidence marker counts.

This does NOT change ingestion outputs; it's a diagnostic tool to validate what the
DOCX actually contains (including tables).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from apps.api.ingest.docx_reader import DocxReader, ParsedDocument, ParsedParagraph
from apps.api.retrieve.normalize_ar import normalize_for_matching


PILLAR_NAME_RE = re.compile(r"(الروحية|العاطفية|الفكرية|البدنية|الاجتماعية)")


def _is_pillar_heading(text: str) -> bool:
    t = text.strip()
    if not t:
        return False
    # same patterns used by extractor (simplified)
    if re.search(r"^(أولا|ثانيا|ثالثا|رابعا|خامسا)[.\s:]*الركيزة", t):
        return True
    if re.search(r"^الركيزة\s+(الروحية|العاطفية|الفكرية|البدنية|الاجتماعية)", t):
        return True
    if re.search(
        r"^(أولا|ثانيا|ثالثا|رابعا|خامسا)?[.\s:]*الحياة\s+(الروحية|العاطفية|الفكرية|البدنية|الاجتماعية)\s+الطيبة",
        t,
    ):
        return True
    return False


def _extract_pillar_name(text: str) -> str:
    m = PILLAR_NAME_RE.search(text)
    if m:
        return f"الحياة {m.group(1)}"
    return text.strip()


def _is_core_value_heading(text: str) -> bool:
    t = text.strip()
    if len(t) > 140 or len(t) < 2:
        return False
    if not re.match(
        r"^(أولا|أولًا|أولاً|ثانيا|ثانيًا|ثالثا|ثالثًا|رابعا|خامسا|سادسا|سابعا|ثامنا|تاسعا|عاشرا|[0-9\u0660-\u0669]+)[.\s:：\-]+",
        t,
    ):
        return False
    # Avoid pillar headings
    if _is_pillar_heading(t):
        return False
    # Avoid headings like "القيم الكلية"
    if normalize_for_matching(t).startswith(normalize_for_matching("القيم")):
        return False
    return True


def _clean_core_value_name(text: str) -> str:
    name = re.sub(
        r"^(أولا|أولًا|أولاً|ثانيا|ثانيًا|ثالثا|ثالثًا|رابعا|خامسا|سادسا|سابعا|ثامنا|تاسعا|عاشرا|[0-9\u0660-\u0669]+)[.\s:：\-]+",
        "",
        text.strip(),
    ).strip()
    name = re.sub(r"^\s*قيمة\s+", "", name).strip()
    if "  " in name:
        name = name.split("  ")[0].strip()
    return name or text.strip()


def _is_sub_value_heading(text: str) -> bool:
    t = text.strip()
    if not t or len(t) > 80:
        return False
    # exclude obvious markers
    norm = normalize_for_matching(t)
    for bad in ["المفهوم", "التأصيل", "الدليل", "الأدلة", "الشواهد", "التعريف"]:
        if norm == normalize_for_matching(bad) or norm.startswith(normalize_for_matching(bad + ":")):
            return False
    # exclude verse/hadith lines
    if t.startswith("﴿") or t.startswith("{") or t.startswith("(") or t.startswith("«"):
        return False
    if t.startswith("قال") or t.startswith("وقال") or t.startswith("يقول") or t.startswith("وفي الحديث"):
        return False
    if t.endswith(":") or t.endswith("："):
        return True
    # short standalone lines
    if len(t.split()) <= 4:
        # avoid list numbering
        if re.match(r"^[\d\u0660-\u0669]+[.\-)\s]+", t):
            return False
        return True
    return False


def _clean_sub_value_name(text: str) -> str:
    return text.strip().rstrip(":：").strip()


def _is_evidence_marker(text: str) -> bool:
    t = text.strip()
    return any(m in t[:50] for m in ["التأصيل", "الأدلة", "الدليل", "الشواهد"])


@dataclass
class SV:
    name_ar: str
    anchor: str


@dataclass
class CV:
    name_ar: str
    anchor: str
    sub_values: list[SV] = field(default_factory=list)


@dataclass
class Pillar:
    name_ar: str
    anchor: str
    core_values: list[CV] = field(default_factory=list)
    evidence_markers: int = 0


def build_outline(doc: ParsedDocument) -> list[Pillar]:
    pillars: list[Pillar] = []
    cur_p: Optional[Pillar] = None
    cur_cv: Optional[CV] = None

    for p in doc.paragraphs:
        t = (p.text or "").strip()
        if not t:
            continue

        if _is_pillar_heading(t):
            cur_p = Pillar(name_ar=_extract_pillar_name(t), anchor=p.source_anchor)
            pillars.append(cur_p)
            cur_cv = None
            continue

        if cur_p and _is_core_value_heading(t):
            name = _clean_core_value_name(t)
            # de-dupe within pillar
            n = normalize_for_matching(name)
            existing = None
            for cv in cur_p.core_values:
                if normalize_for_matching(cv.name_ar) == n:
                    existing = cv
                    break
            if existing:
                cur_cv = existing
            else:
                cur_cv = CV(name_ar=name, anchor=p.source_anchor)
                cur_p.core_values.append(cur_cv)
            continue

        if cur_cv and _is_sub_value_heading(t):
            name = _clean_sub_value_name(t)
            n = normalize_for_matching(name)
            if not any(normalize_for_matching(sv.name_ar) == n for sv in cur_cv.sub_values):
                cur_cv.sub_values.append(SV(name_ar=name, anchor=p.source_anchor))
            continue

        if cur_p and _is_evidence_marker(t):
            cur_p.evidence_markers += 1

    return pillars


def main() -> None:
    docx = Path("docs/source/framework_2025-10_v1.docx")
    parsed = DocxReader().read(docx)

    pillars = build_outline(parsed)
    out = {
        "doc_name": parsed.doc_name,
        "doc_hash": parsed.doc_hash,
        "total_paragraphs_including_tables": parsed.total_paragraphs,
        "pillars": [
            {
                "name_ar": p.name_ar,
                "anchor": p.anchor,
                "core_values": [
                    {
                        "name_ar": cv.name_ar,
                        "anchor": cv.anchor,
                        "sub_values": [{"name_ar": sv.name_ar, "anchor": sv.anchor} for sv in cv.sub_values],
                    }
                    for cv in p.core_values
                ],
                "evidence_markers": p.evidence_markers,
            }
            for p in pillars
        ],
        "counts": {
            "pillars": len(pillars),
            "core_values": sum(len(p.core_values) for p in pillars),
            "sub_values": sum(len(cv.sub_values) for p in pillars for cv in p.core_values),
        },
    }

    derived = Path("data/derived")
    derived.mkdir(parents=True, exist_ok=True)
    out_path = derived / "framework_2025-10_v1.analysis.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    print("DOC:", parsed.doc_name)
    print("HASH:", parsed.doc_hash[:16], "…")
    print("TOTAL_PARAS(incl tables):", parsed.total_paragraphs)
    print("COUNTS:", out["counts"])
    for p in pillars:
        print(f"- PILLAR: {p.name_ar} core={len(p.core_values)} evidence_markers={p.evidence_markers}")
        for cv in p.core_values:
            print(f"  - CV: {cv.name_ar} sub={len(cv.sub_values)}")
    print("WROTE:", str(out_path))


if __name__ == "__main__":
    main()


