"""Rule extractor implementation (paragraph processing).

Reason: split `rule_extractor.py` into <500 LOC modules.
"""

from __future__ import annotations

import re

from apps.api.ingest.docx_reader import ParsedParagraph
from apps.api.retrieve.normalize_ar import normalize_for_matching

from apps.api.ingest.rule_extractor_types import ExtractorState


class _RuleExtractorProcessMixin:
    def _process_paragraph(self, para: ParsedParagraph, doc_hash: str, source_doc: str) -> None:
        """Process a single paragraph through the state machine."""
        text = para.text.strip()

        # In pillar sub-values table mode, empty cells must be counted to keep column alignment.
        if not text:
            if self.state == ExtractorState.IN_SUB_VALUES_LIST and self._subvalues_table_mode:
                self._subvalues_table_col = (self._subvalues_table_col + 1) % max(
                    1, len(self._subvalues_table_core_order) or 3
                )
            return

        normalized = normalize_for_matching(text)

        if (
            self.current_sub_value
            and self.state
            in (
                ExtractorState.IN_SUB_VALUE,
                ExtractorState.IN_DEFINITION,
                ExtractorState.IN_EVIDENCE,
            )
            and ":" in text
        ):
            left = text.split(":", 1)[0].strip()
            if re.match(r"^[\d\u0660-\u0669]+", left):
                pass
            else:
                if (
                    1 < len(left) <= 20
                    and normalize_for_matching(left) in normalize_for_matching(self.current_sub_value.name_ar)
                    and normalize_for_matching(left) != normalize_for_matching(self.current_sub_value.name_ar)
                ):
                    self._continue_current_section(para, doc_hash, source_doc, text)
                    return

        if (
            self.state == ExtractorState.IN_DEFINITION
            and self.current_sub_value
            and ":" in text
            and normalize_for_matching(text.split(":", 1)[0].strip()) == normalize_for_matching(self.current_sub_value.name_ar)
        ):
            self._continue_current_section(para, doc_hash, source_doc, text)
            return

        if (
            getattr(para, "style", "") == "OCR_USER_IMAGE"
            and self.current_core_value
            and ":" in text
            and str(getattr(para, "source_anchor", "")).endswith("_ln0")
        ):
            left, right = text.split(":", 1)
            left_s = left.strip()
            right_s = right.strip()
            if re.match(
                r"^(أولا|أولًا|أولاً|ثانيا|ثانيًا|ثالثا|ثالثًا|رابعا|خامسا|سادسا|سابعا|ثامنا|تاسعا|عاشرا)\b",
                left_s,
            ):
                pass
            else:
                if (
                    left_s
                    and right_s
                    and len(left_s) <= 40
                    and len(left_s.split()) <= 4
                    and len(right_s) >= 8
                    and normalize_for_matching(left_s)
                    not in {
                        normalize_for_matching("المفهوم"),
                        normalize_for_matching("التأصيل"),
                        normalize_for_matching("التعريف"),
                        normalize_for_matching("الدليل"),
                    }
                    and not left_s.startswith(("قال", "وقال", "يقول"))
                ):
                    self._finalize_definition_and_evidence()
                    self._start_sub_value_from_heading(para, doc_hash, source_doc, left_s)
                    self._handle_definition_start(para, doc_hash, source_doc, "المفهوم:")
                    self._continue_current_section(para, doc_hash, source_doc, right_s)
                    return

        if self._is_pillars_list_marker(normalized):
            self._handle_pillars_list()
            return

        if self._is_pillar_marker(text):
            self._handle_pillar_start(para, doc_hash, source_doc, text)
            return

        if self._is_core_values_list_marker(normalized):
            self._handle_core_values_list()
            return

        if self._is_sub_values_list_marker(normalized):
            self._handle_sub_values_list()
            return

        if self.current_pillar and self._skeleton_core_values_by_pillar:
            candidate = text.strip().rstrip(":：").strip()
            if (
                candidate
                and len(candidate) <= 40
                and len(candidate.split()) <= 4
                and not any(ch in candidate for ch in ["،", "“", "”", "\"", ".", "…", "؛"])
                and self._core_value_allowed_for_current_pillar(candidate)
            ):
                self._finalize_definition_and_evidence()
                self._subvalues_table_mode = False
                self._start_core_value_from_heading(para, doc_hash, source_doc, candidate)
                return

        if self.current_core_value and re.match(r"^[\d\u0660-\u0669]", text.strip()):
            if self._is_sub_value_heading(text):
                self._finalize_definition_and_evidence()
                self._start_sub_value_from_heading(para, doc_hash, source_doc, text)
                return

        if self._is_core_value_heading(text) and self.current_pillar:
            cleaned = self._clean_core_value_heading_name(text)
            if self._core_value_allowed_for_current_pillar(cleaned):
                self._finalize_definition_and_evidence()
                self._subvalues_table_mode = False
                self._start_core_value_from_heading(para, doc_hash, source_doc, text)
                return

        if (
            self.current_core_value
            and self.current_core_value.sub_values
            and self.state
            in (
                ExtractorState.IN_CORE_VALUE,
                ExtractorState.IN_SUB_VALUES_LIST,
                ExtractorState.IN_SUB_VALUE,
                ExtractorState.IN_DEFINITION,
                ExtractorState.IN_EVIDENCE,
            )
        ):
            candidate = self._clean_value_name(text.strip().rstrip(":：").strip())
            if candidate and len(candidate) <= 80:
                cand_n = normalize_for_matching(candidate)
                for sv in self.current_core_value.sub_values:
                    if normalize_for_matching(sv.name_ar) == cand_n:
                        if (
                            self.state == ExtractorState.IN_DEFINITION
                            and self.current_sub_value
                            and normalize_for_matching(self.current_sub_value.name_ar) == cand_n
                        ):
                            break
                        self._finalize_definition_and_evidence()
                        self._start_sub_value_from_heading(para, doc_hash, source_doc, candidate)
                        return

        if self._is_sub_value_heading(text) and self.current_core_value:
            self._finalize_definition_and_evidence()
            self._start_sub_value_from_heading(para, doc_hash, source_doc, text)
            return

        if self._is_definition_marker(text):
            self._handle_definition_start(para, doc_hash, source_doc, text)
            return

        if self._is_evidence_marker(text):
            self._handle_evidence_start(para, doc_hash, source_doc, text)
            return

        self._continue_current_section(para, doc_hash, source_doc, text)

