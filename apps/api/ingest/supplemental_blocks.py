"""
Supplemental OCR Blocks → anchored text_blocks

Why:
- Users may provide screenshots to ensure nothing is missed.
- Not every screenshot is a (definition/evidence) block that the rule extractor will attach.
- We still must persist what the user sent with stable anchors for audit and proof.

Approach:
- Read supplemental OCR JSON payloads from `data/derived/supplemental_ocr/<doc_hash>/*.json`
- Resolve payload context (pillar/core/sub) to canonical IDs in the current canonical JSON
- Emit `supplemental_text_blocks` entries that the DB loader writes into `text_block`

Notes:
- This does NOT overwrite entity `source_anchor` (DOCX remains source-of-truth).
- These blocks are stored as `block_type="supplemental_ocr"` for traceability.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from apps.api.retrieve.normalize_ar import normalize_for_matching


def _supp_dir() -> Path:
    return Path(os.getenv("SUPPLEMENTAL_OCR_DIR", "data/derived/supplemental_ocr"))


@dataclass(frozen=True)
class SupplementalTextBlock:
    entity_type: str  # pillar | core_value | sub_value
    entity_id: str
    block_type: str  # supplemental_ocr
    text_ar: str
    source_anchor: dict[str, Any]  # {"source_anchor": "..."}
    context: dict[str, str]
    image_sha256: str
    filename: str


def _iter_payload_files(doc_hash: str) -> list[Path]:
    d = _supp_dir() / doc_hash
    if not d.exists():
        return []
    return sorted(d.glob("*.json"))


def _load_payload(path: Path) -> Optional[dict[str, Any]]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _build_name_index(canonical: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """
    Build fast lookup indices by normalized Arabic names.

    Returns:
        {
          "pillar": {norm_name: pillar_dict},
          "core_value": {(pillar_id, norm_name): cv_dict},
          "sub_value": {(cv_id, norm_name): sv_dict},
        }
    """
    pillars = canonical.get("pillars") or []
    pillar_by = {normalize_for_matching(p.get("name_ar", "")): p for p in pillars if p.get("name_ar")}

    cv_by: dict[tuple[str, str], dict[str, Any]] = {}
    sv_by: dict[tuple[str, str], dict[str, Any]] = {}

    for p in pillars:
        pid = str(p.get("id") or "")
        for cv in (p.get("core_values") or []):
            cv_key = (pid, normalize_for_matching(cv.get("name_ar", "")))
            if pid and cv.get("name_ar"):
                cv_by[cv_key] = cv
            cvid = str(cv.get("id") or "")
            for sv in (cv.get("sub_values") or []):
                sv_key = (cvid, normalize_for_matching(sv.get("name_ar", "")))
                if cvid and sv.get("name_ar"):
                    sv_by[sv_key] = sv

    return {"pillar": pillar_by, "core_value": cv_by, "sub_value": sv_by}


def build_supplemental_text_blocks_for_canonical(
    canonical: dict[str, Any],
) -> list[dict[str, Any]]:
    """
    Build supplemental OCR text blocks for canonical JSON.

    Expected payload shape (saved by supplemental_ocr):
    - image_sha256, filename, lines[], context{pillar_name_ar, core_value_name_ar, sub_value_name_ar}

    Returns a list of dicts:
    - entity_type, entity_id, block_type, text_ar, source_anchor, context, image_sha256, filename
    """
    meta = canonical.get("meta") or {}
    doc_hash = str(meta.get("source_file_hash") or "")
    if not doc_hash:
        return []

    idx = _build_name_index(canonical)
    out: list[dict[str, Any]] = []

    for p in _iter_payload_files(doc_hash):
        payload = _load_payload(p)
        if not payload:
            continue

        img_sha = str(payload.get("image_sha256") or "").strip()
        filename = str(payload.get("filename") or p.name).strip()
        lines = payload.get("lines") or []
        ctx = payload.get("context") or {}

        pillar_name = str(ctx.get("pillar_name_ar") or "").strip()
        core_name = str(ctx.get("core_value_name_ar") or "").strip()
        sub_name = str(ctx.get("sub_value_name_ar") or "").strip()

        if not img_sha or not isinstance(lines, list):
            continue

        text_lines = [str(x).strip() for x in lines if str(x).strip()]
        if not text_lines:
            continue

        # Resolve entity target from context (most specific wins).
        entity_type: Optional[str] = None
        entity_id: Optional[str] = None

        pillar = None
        if pillar_name:
            pillar = idx["pillar"].get(normalize_for_matching(pillar_name))

        if sub_name and core_name and pillar:
            cv = idx["core_value"].get((str(pillar.get("id")), normalize_for_matching(core_name)))
            if cv:
                sv = idx["sub_value"].get((str(cv.get("id")), normalize_for_matching(sub_name)))
                if sv:
                    entity_type = "sub_value"
                    entity_id = str(sv.get("id"))
        if not entity_id and core_name and pillar:
            cv = idx["core_value"].get((str(pillar.get("id")), normalize_for_matching(core_name)))
            if cv:
                entity_type = "core_value"
                entity_id = str(cv.get("id"))
        if not entity_id and pillar:
            entity_type = "pillar"
            entity_id = str(pillar.get("id"))

        # If we cannot resolve context, store nothing (no safe place to attach).
        if not entity_type or not entity_id:
            continue

        # Anchor to the first OCR line of that screenshot payload.
        source_anchor = {"source_anchor": f"userimg_{img_sha[:12]}_ln0"}
        out.append(
            SupplementalTextBlock(
                entity_type=entity_type,
                entity_id=entity_id,
                block_type="supplemental_ocr",
                text_ar="\n".join(text_lines),
                source_anchor=source_anchor,
                context={
                    "pillar_name_ar": pillar_name,
                    "core_value_name_ar": core_name,
                    "sub_value_name_ar": sub_name,
                },
                image_sha256=img_sha,
                filename=filename,
            ).__dict__
        )

    return out


