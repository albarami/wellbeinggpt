"""
Canonical JSON Export Module

Converts extraction results to canonical JSON format for:
1. Debugging and inspection
2. Loading into database
3. Versioning and archiving
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from apps.api.ingest.rule_extractor import (
    ExtractionResult,
    ExtractedPillar,
    ExtractedCoreValue,
    ExtractedSubValue,
    ExtractedDefinition,
)
from apps.api.ingest.evidence_parser import ParsedEvidence, ParsedQuranRef, ParsedHadithRef


def extraction_to_canonical_json(
    result: ExtractionResult,
    evidence_map: Optional[dict[str, ParsedEvidence]] = None,
) -> dict[str, Any]:
    """
    Convert an ExtractionResult to canonical JSON format.

    Args:
        result: The extraction result.
        evidence_map: Optional mapping of entity IDs to parsed evidence.

    Returns:
        Dictionary in canonical JSON format.
    """
    return {
        "meta": {
            "source_doc_id": result.source_doc_id,
            "source_file_hash": result.source_file_hash,
            "source_doc": result.source_doc,
            "source_hash": result.source_file_hash,
            "framework_version": result.framework_version,
            "extracted_at": datetime.utcnow().isoformat(),
            "stats": {
                "total_pillars": len(result.pillars),
                "total_core_values": result.total_core_values,
                "total_sub_values": result.total_sub_values,
                "total_evidence": result.total_evidence,
            },
            "validation_errors": result.validation_errors,
            "warnings": result.warnings,
        },
        "pillars": [
            _pillar_to_dict(p, evidence_map) for p in result.pillars
        ],
    }


def _pillar_to_dict(
    pillar: ExtractedPillar,
    evidence_map: Optional[dict[str, ParsedEvidence]] = None,
) -> dict[str, Any]:
    """Convert a pillar to dictionary."""
    return {
        "id": pillar.id,
        "name_ar": pillar.name_ar,
        "name_en": None,  # Nullable per MVP policy
        "description_ar": pillar.description_ar,
        "source_doc": pillar.source_doc,
        "source_hash": pillar.source_hash,
        "source_anchor": pillar.source_anchor,
        "raw_text": pillar.raw_text,
        "para_index": pillar.para_index,
        "core_values": [
            _core_value_to_dict(cv, evidence_map)
            for cv in pillar.core_values
        ],
    }


def _core_value_to_dict(
    cv: ExtractedCoreValue,
    evidence_map: Optional[dict[str, ParsedEvidence]] = None,
) -> dict[str, Any]:
    """Convert a core value to dictionary."""
    return {
        "id": cv.id,
        "name_ar": cv.name_ar,
        "name_en": None,
        "definition": _definition_to_dict(cv.definition) if cv.definition else None,
        "source_doc": cv.source_doc,
        "source_hash": cv.source_hash,
        "source_anchor": cv.source_anchor,
        "raw_text": cv.raw_text,
        "para_index": cv.para_index,
        "evidence": _evidence_list_to_dict(cv.evidence) if cv.evidence else [],
        "sub_values": [
            _sub_value_to_dict(sv, evidence_map)
            for sv in cv.sub_values
        ],
    }


def _sub_value_to_dict(
    sv: ExtractedSubValue,
    evidence_map: Optional[dict[str, ParsedEvidence]] = None,
) -> dict[str, Any]:
    """Convert a sub-value to dictionary."""
    return {
        "id": sv.id,
        "name_ar": sv.name_ar,
        "name_en": None,
        "definition": _definition_to_dict(sv.definition) if sv.definition else None,
        "source_doc": sv.source_doc,
        "source_hash": sv.source_hash,
        "source_anchor": sv.source_anchor,
        "raw_text": sv.raw_text,
        "para_index": sv.para_index,
        "evidence": _evidence_list_to_dict(sv.evidence) if sv.evidence else [],
    }


def _definition_to_dict(definition: ExtractedDefinition) -> dict[str, Any]:
    """Convert a definition to dictionary."""
    return {
        "text_ar": definition.text_ar,
        "text_en": None,
        "source_doc": definition.source_doc,
        "source_hash": definition.source_hash,
        "source_anchor": definition.source_anchor,
        "raw_text": definition.raw_text,
        "para_indices": definition.para_indices,
    }


def _evidence_list_to_dict(evidence_list: list) -> list[dict[str, Any]]:
    """Convert a list of evidence to dictionaries."""
    return [_evidence_to_dict(e) for e in evidence_list]


def _evidence_to_dict(evidence) -> dict[str, Any]:
    """Convert evidence to dictionary."""
    if hasattr(evidence, "evidence_type"):
        return {
            "evidence_type": evidence.evidence_type,
            "ref_raw": evidence.ref_raw,
            "text_ar": evidence.text_ar,
            "source_doc": evidence.source_doc,
            "source_hash": evidence.source_hash,
            "source_anchor": evidence.source_anchor,
            "raw_text": evidence.raw_text,
            "para_index": evidence.para_index,
        }
    return {}


def save_canonical_json(
    data: dict[str, Any],
    output_path: str | Path,
    indent: int = 2,
) -> None:
    """
    Save canonical JSON to a file.

    Args:
        data: The canonical JSON data.
        output_path: Path to save the file.
        indent: JSON indentation level.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=indent)


def load_canonical_json(input_path: str | Path) -> dict[str, Any]:
    """
    Load canonical JSON from a file.

    Args:
        input_path: Path to the JSON file.

    Returns:
        The loaded data.
    """
    path = Path(input_path)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

