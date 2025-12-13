"""
Supplemental Structure Enforcer

Why:
- The user provides screenshots of the authoritative sub-value tables.
- The DOCX can contain headings that resemble value names; OCR can also introduce artifacts.
- For enterprise-grade correctness, we must align the extracted hierarchy with the authoritative table.

Scope:
- Currently applied to the Social pillar only (الحياة الاجتماعية), because we have explicit
  table screenshots for its core values.

This module reads supplemental OCR payloads whose filenames include '*_structure.png' and
filters canonical core_value.sub_values to match those lists.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

from apps.api.retrieve.normalize_ar import normalize_for_matching


def _supp_dir() -> Path:
    return Path(os.getenv("SUPPLEMENTAL_OCR_DIR", "data/derived/supplemental_ocr"))


def _clean_list_name(text: str) -> str:
    t = (text or "").strip()
    # strip common numbering patterns: "1)" / "1." / "(1)" etc
    t = re.sub(r"^\(\s*[\d\u0660-\u0669]+\s*\)\s*", "", t)
    t = re.sub(r"^[\d\u0660-\u0669]+[.\-)\s]+", "", t)
    t = re.sub(r"^[•\\-–—]\\s*", "", t)
    t = t.strip().rstrip(":：").strip()
    # normalize parentheses qualifier used in table
    t = t.replace("(في الأدوار الاجتماعية)", "في الأدوار الاجتماعية")
    t = t.replace("(قيمتان متلازمتان)", "(قيمة متلازمة)")
    return t


def enforce_social_structure_from_supplemental_tables(canonical: dict[str, Any]) -> dict[str, Any]:
    """
    Filter social pillar sub-values using supplemental '*_structure.png' OCR payloads.

    This keeps canonical IDs (we only filter existing items by name).
    """
    meta = canonical.get("meta") or {}
    doc_hash = str(meta.get("source_file_hash") or "")
    if not doc_hash:
        return canonical

    d = _supp_dir() / doc_hash
    if not d.exists():
        return canonical

    # Build allowed lists keyed by normalized core name.
    allowed_by_core: dict[str, set[str]] = {}
    for p in d.glob("*.json"):
        try:
            payload = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        filename = str(payload.get("filename") or "")
        if "structure" not in filename:
            continue
        ctx = payload.get("context") or {}
        pillar = str(ctx.get("pillar_name_ar") or "").strip()
        core = str(ctx.get("core_value_name_ar") or "").strip()
        if normalize_for_matching(pillar) != normalize_for_matching("الحياة الاجتماعية"):
            continue
        if not core:
            continue
        lines = payload.get("lines") or []
        names = []
        for ln in lines:
            nm = _clean_list_name(str(ln))
            if nm and len(nm) <= 80 and normalize_for_matching(nm) not in {
                normalize_for_matching("القيم"),
                normalize_for_matching("الأحفاد"),
            }:
                names.append(nm)
        if not names:
            continue
        allowed_by_core.setdefault(normalize_for_matching(core), set()).update(
            normalize_for_matching(x) for x in names
        )

    if not allowed_by_core:
        return canonical

    for pillar in canonical.get("pillars", []):
        if normalize_for_matching(pillar.get("name_ar", "")) != normalize_for_matching("الحياة الاجتماعية"):
            continue
        for cv in pillar.get("core_values", []):
            allowed = allowed_by_core.get(normalize_for_matching(cv.get("name_ar", "")))
            if not allowed:
                continue
            def is_allowed(name_ar: str) -> bool:
                n = normalize_for_matching(name_ar or "")
                if n in allowed:
                    return True
                # Allow prefix matches for cases where the table lists a base name and
                # the canonical sub-value includes a qualifier in parentheses.
                return any(n.startswith(a) for a in allowed)

            cv["sub_values"] = [sv for sv in (cv.get("sub_values") or []) if is_allowed(sv.get("name_ar", ""))]

    return canonical


