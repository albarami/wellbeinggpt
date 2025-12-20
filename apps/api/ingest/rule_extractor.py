"""Rule-Based Extractor Module.

This file is intentionally kept <500 LOC.
Implementation is split across:
- `rule_extractor_types.py` (dataclasses + enum)
- `rule_extractor_data.py` (marker constants)
- `rule_extractor_impl_*.py` (mixins + assembly)

Public API remains stable: import `RuleExtractor` and `ExtractionResult` from here.
"""

from __future__ import annotations

from apps.api.ingest.rule_extractor_impl import RuleExtractor
from apps.api.ingest.rule_extractor_types import (
    ExtractedCoreValue,
    ExtractedDefinition,
    ExtractedEvidence,
    ExtractedPillar,
    ExtractedSubValue,
    ExtractionResult,
    ExtractorState,
)

__all__ = [
    "ExtractorState",
    "ExtractedEvidence",
    "ExtractedDefinition",
    "ExtractedSubValue",
    "ExtractedCoreValue",
    "ExtractedPillar",
    "ExtractionResult",
    "RuleExtractor",
]
