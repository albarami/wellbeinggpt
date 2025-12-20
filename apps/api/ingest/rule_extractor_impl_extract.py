"""Rule extractor implementation (init + extract entrypoint).

Reason: split `rule_extractor.py` into <500 LOC modules.
"""

from __future__ import annotations

from typing import Optional

from apps.api.ingest.docx_reader import ParsedDocument, ParsedParagraph
from apps.api.retrieve.normalize_ar import normalize_for_matching

from apps.api.ingest.rule_extractor_types import (
    ExtractedCoreValue,
    ExtractedPillar,
    ExtractedSubValue,
    ExtractionResult,
    ExtractorState,
)


class _RuleExtractorInitExtractMixin:
    def __init__(self, framework_version: str = "2025-10"):
        self.framework_version = framework_version
        self._reset()

    def _reset(self) -> None:
        self.state = ExtractorState.INITIAL
        self.current_pillar: Optional[ExtractedPillar] = None
        self.current_core_value: Optional[ExtractedCoreValue] = None
        self.current_sub_value: Optional[ExtractedSubValue] = None
        self.current_definition_paras: list[dict] = []
        self.current_evidence_paras: list[dict] = []
        self.current_block_refs: list[dict] = []

        self.pillar_counter = 0
        self.core_value_counter = 0
        self.sub_value_counter = 0

        self.pillars: list[ExtractedPillar] = []
        self.validation_errors: list[str] = []
        self.warnings: list[str] = []

        self._skeleton_core_values_by_pillar: dict[str, list[str]] = {}
        self._skip_para_indices: set[int] = set()
        self._pillars_by_canonical: dict[str, ExtractedPillar] = {}
        self._allowed_subvalues_by_core: dict[str, set[str]] = {}

        self._subvalues_table_mode: bool = False
        self._subvalues_table_core_order: list[ExtractedCoreValue] = []
        self._subvalues_table_col: int = 0

    def extract(self, doc: ParsedDocument) -> ExtractionResult:
        self._reset()

        source_doc_id = f"DOC_{doc.doc_hash[:16]}"
        source_doc = f"docs/source/{doc.doc_name}"

        self._extract_framework_skeleton(doc)

        for para in doc.paragraphs:
            if para.para_index in self._skip_para_indices:
                continue
            self._process_paragraph(para, doc.doc_hash, source_doc)

        self._finalize_current_sections(doc.doc_hash, source_doc)
        self._apply_allowed_subvalue_filters()

        total_cv = sum(len(p.core_values) for p in self.pillars)
        total_sv = sum(len(cv.sub_values) for p in self.pillars for cv in p.core_values)
        total_ev = sum(
            len(cv.evidence) + sum(len(sv.evidence) for sv in cv.sub_values) for p in self.pillars for cv in p.core_values
        )

        return ExtractionResult(
            source_doc_id=source_doc_id,
            source_file_hash=doc.doc_hash,
            source_doc=source_doc,
            framework_version=self.framework_version,
            pillars=self.pillars,
            validation_errors=self.validation_errors,
            warnings=self.warnings,
            total_core_values=total_cv,
            total_sub_values=total_sv,
            total_evidence=total_ev,
        )

    def _apply_allowed_subvalue_filters(self) -> None:
        if not self._skeleton_core_values_by_pillar:
            return
        for p in self.pillars:
            if normalize_for_matching(p.name_ar) != normalize_for_matching("الحياة الاجتماعية"):
                continue
            for cv in p.core_values:
                allowed = self._allowed_subvalues_by_core.get(cv.id)
                if not allowed:
                    continue
                cv.sub_values = [sv for sv in cv.sub_values if normalize_for_matching(sv.name_ar) in allowed]

