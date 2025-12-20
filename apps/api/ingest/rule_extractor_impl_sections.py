"""Rule extractor implementation (markers + section handlers + skeleton scan).

Reason: split `rule_extractor.py` into <500 LOC modules.
"""

from __future__ import annotations

import re
from typing import Optional

from apps.api.ingest.docx_reader import ParsedDocument, ParsedParagraph
from apps.api.retrieve.normalize_ar import normalize_for_matching

from apps.api.ingest.rule_extractor_data import (
    DEFINITION_MARKERS,
    EVIDENCE_MARKERS,
    PILLAR_MARKERS,
    PILLARS_LIST_MARKERS,
)
from apps.api.ingest.rule_extractor_types import ExtractedCoreValue, ExtractedPillar, ExtractorState


class _RuleExtractorSectionsMixin:
    def _is_pillars_list_marker(self, normalized_text: str) -> bool:
        for marker in PILLARS_LIST_MARKERS:
            if normalize_for_matching(marker) in normalized_text:
                return True
        return False

    def _is_pillar_marker(self, text: str) -> bool:
        for pattern in PILLAR_MARKERS:
            if pattern.search(text):
                return True
        return False

    def _is_core_values_list_marker(self, normalized_text: str) -> bool:
        if not normalized_text.startswith(normalize_for_matching("القيم")):
            return False
        return normalize_for_matching("الكلية") in normalized_text or normalize_for_matching("الأمهات") in normalized_text

    def _is_sub_values_list_marker(self, normalized_text: str) -> bool:
        if normalize_for_matching("الأحفاد") in normalized_text and len(normalized_text) <= 20:
            return True
        if not normalized_text.startswith(normalize_for_matching("القيم")):
            return False
        return normalize_for_matching("الجزئية") in normalized_text or normalize_for_matching("الأحفاد") in normalized_text

    def _is_definition_marker(self, text: str) -> bool:
        n = normalize_for_matching(text[:80])
        for marker in DEFINITION_MARKERS:
            if normalize_for_matching(marker) in n:
                return True
        return False

    def _is_evidence_marker(self, text: str) -> bool:
        n = normalize_for_matching(text[:80])
        for marker in EVIDENCE_MARKERS:
            if normalize_for_matching(marker) in n:
                return True
        return False

    def _handle_pillars_list(self) -> None:
        self._finalize_definition_and_evidence()
        self.state = ExtractorState.IN_PILLARS_LIST

    def _handle_pillar_start(self, para: ParsedParagraph, doc_hash: str, source_doc: str, text: str) -> None:
        self._finalize_definition_and_evidence()
        name = self._canonicalize_pillar_name(self._extract_pillar_name(text))

        if self._pillars_by_canonical:
            if name not in self._pillars_by_canonical:
                return
            self.current_pillar = self._pillars_by_canonical[name]
            if self.current_pillar.source_anchor.startswith("para_") and self.current_pillar.raw_text:
                if ("الطيبة" in (self.current_pillar.raw_text or "")) and (
                    "الركيزة" in text or "أولا" in text or "ثانيا" in text
                ):
                    self.current_pillar.source_anchor = para.source_anchor
                    self.current_pillar.raw_text = text
                    self.current_pillar.para_index = para.para_index
            else:
                self.current_pillar.source_anchor = para.source_anchor
                self.current_pillar.raw_text = text
                self.current_pillar.para_index = para.para_index

            self.current_core_value = None
            self.current_sub_value = None
            self.state = ExtractorState.IN_PILLAR
            return

        self.pillar_counter += 1
        pillar_id = f"P{self.pillar_counter:03d}"
        self.current_pillar = ExtractedPillar(
            id=pillar_id,
            name_ar=name,
            source_doc=source_doc,
            source_hash=doc_hash,
            source_anchor=para.source_anchor,
            raw_text=text,
            para_index=para.para_index,
        )
        self.state = ExtractorState.IN_PILLAR

    def _extract_pillar_name(self, text: str) -> str:
        name = re.sub(r"^(أولا|ثانيا|ثالثا|رابعا|خامسا)[.\\s:]*", "", text)
        name = re.sub(r"^الركيزة\\s*", "", name)
        name = name.strip(" :.،")
        return name if name else text

    def _extract_framework_skeleton(self, doc: ParsedDocument) -> None:
        marker_norm = normalize_for_matching("ركائز الحياة الطيبة")
        marker_pi: Optional[int] = None
        for p in doc.paragraphs:
            if marker_norm in normalize_for_matching(p.text or ""):
                marker_pi = p.para_index
                break
        if marker_pi is None:
            return

        window = [p for p in doc.paragraphs if p.para_index >= marker_pi and (p.text or "").strip()]
        if len(window) < 30:
            return

        def is_header_cell(txt: str) -> bool:
            n = normalize_for_matching(txt)
            if normalize_for_matching("الحياة") not in n:
                return False
            if normalize_for_matching("الطيبة") not in n:
                return False
            return any(
                normalize_for_matching(x) in n for x in ["الروحية", "العاطفية", "الفكرية", "البدنية", "الاجتماعية"]
            )

        header_pos: Optional[int] = None
        for i in range(min(len(window), 120)):
            if is_header_cell(window[i].text):
                if i + 4 < len(window) and all(is_header_cell(window[i + j].text) for j in range(5)):
                    header_pos = i
                    break
        if header_pos is None:
            return

        headers = [self._canonicalize_pillar_name(window[header_pos + j].text) for j in range(5)]
        core_start = header_pos + 5
        if core_start + 15 > len(window):
            return

        mapping: dict[str, list[str]] = {h: [] for h in headers}
        for row_idx in range(3):
            for col_idx in range(5):
                cell_txt = (window[core_start + row_idx * 5 + col_idx].text or "").strip()
                if cell_txt:
                    mapping[headers[col_idx]].append(cell_txt)

        if not all(len(v) >= 3 for v in mapping.values()):
            self.warnings.append("Skeleton table detected but incomplete; core value gating disabled.")
            return

        self._skeleton_core_values_by_pillar = {k: v[:3] for k, v in mapping.items()}

        to_skip = [window[0]] + window[header_pos : header_pos + 5 + 15]
        self._skip_para_indices = {p.para_index for p in to_skip}

        self._pillars_by_canonical = {}
        self.pillars = []
        self.pillar_counter = 0
        for i, h in enumerate(headers):
            pid = f"P{i+1:03d}"
            pillar_obj = ExtractedPillar(
                id=pid,
                name_ar=h,
                source_doc=f"docs/source/{doc.doc_name}",
                source_hash=doc.doc_hash,
                source_anchor=window[header_pos + i].source_anchor,
                raw_text=window[header_pos + i].text,
                para_index=window[header_pos + i].para_index,
            )
            self._pillars_by_canonical[h] = pillar_obj
            self.pillars.append(pillar_obj)

    def _handle_core_values_list(self) -> None:
        self._finalize_definition_and_evidence()
        self.state = ExtractorState.IN_CORE_VALUES_LIST

    def _handle_sub_values_list(self) -> None:
        self._finalize_definition_and_evidence()
        self.state = ExtractorState.IN_SUB_VALUES_LIST

        self._subvalues_table_mode = False
        self._subvalues_table_core_order = []
        self._subvalues_table_col = 0
        if self._skeleton_core_values_by_pillar and self.current_pillar and len(self.current_pillar.core_values) >= 3:
            allowed = self._skeleton_core_values_by_pillar.get(self.current_pillar.name_ar) or []
            if len(allowed) >= 3:
                ordered: list[ExtractedCoreValue] = []
                for a in allowed[:3]:
                    a_n = normalize_for_matching(a)
                    for cv in self.current_pillar.core_values:
                        if normalize_for_matching(cv.name_ar) == a_n:
                            ordered.append(cv)
                            break
                if len(ordered) == 3:
                    self._subvalues_table_mode = True
                    self._subvalues_table_core_order = ordered
                    self._subvalues_table_col = 0

    def _handle_definition_start(self, para: ParsedParagraph, doc_hash: str, source_doc: str, text: str) -> None:
        self._finalize_definition_and_evidence()
        self.current_block_refs = []

        raw_line = text
        clean = text
        for marker in DEFINITION_MARKERS:
            clean = clean.replace(marker, "").strip()
        if clean.strip() in {":", "："}:
            clean = ""

        self.current_definition_paras = [
            {"raw": raw_line, "clean": clean, "i": para.para_index, "a": para.source_anchor, "doc": source_doc, "hash": doc_hash}
        ]
        self.state = ExtractorState.IN_DEFINITION

    def _handle_evidence_start(self, para: ParsedParagraph, doc_hash: str, source_doc: str, text: str) -> None:
        self._finalize_definition(doc_hash)
        self.current_block_refs = []

        raw_line = text
        clean = text
        for marker in EVIDENCE_MARKERS:
            clean = clean.replace(marker, "").strip()

        self.current_evidence_paras = [
            {"raw": raw_line, "clean": clean, "i": para.para_index, "a": para.source_anchor, "doc": source_doc, "hash": doc_hash}
        ]
        self.state = ExtractorState.IN_EVIDENCE

    def _continue_current_section(self, para: ParsedParagraph, doc_hash: str, source_doc: str, text: str) -> None:
        if re.match(r"^\\d+\\s+.+", text.strip()) and (self.state in (ExtractorState.IN_DEFINITION, ExtractorState.IN_EVIDENCE)):
            self.current_block_refs.append({"type": "book", "ref": text.strip()})
            return

        if self.state == ExtractorState.IN_DEFINITION:
            self.current_definition_paras.append({"raw": text, "clean": text, "i": para.para_index, "a": para.source_anchor, "doc": source_doc, "hash": doc_hash})
        elif self.state == ExtractorState.IN_EVIDENCE:
            self.current_evidence_paras.append({"raw": text, "clean": text, "i": para.para_index, "a": para.source_anchor, "doc": source_doc, "hash": doc_hash})
        elif self.state == ExtractorState.IN_CORE_VALUES_LIST:
            self._try_add_core_value(para, doc_hash, source_doc, text)
        elif self.state == ExtractorState.IN_SUB_VALUES_LIST:
            if self._subvalues_table_mode and self._subvalues_table_core_order:
                if self._is_sub_values_list_marker(normalize_for_matching(text)):
                    self._subvalues_table_col = 0
                    return

                t = text.strip()
                if (
                    t
                    and len(t) <= 80
                    and not self._is_definition_marker(t)
                    and not self._is_evidence_marker(t)
                    and not (t.startswith("﴿") or t.startswith("{") or t.startswith("(") or t.startswith("«"))
                ):
                    cv = self._subvalues_table_core_order[self._subvalues_table_col]
                    cleaned = self._clean_value_name(t)
                    if not cleaned:
                        self._subvalues_table_col = (self._subvalues_table_col + 1) % len(self._subvalues_table_core_order)
                        return
                    prev_cv = self.current_core_value
                    prev_sv = self.current_sub_value
                    prev_state = self.state
                    self.current_core_value = cv
                    self._add_sub_value_name(para, doc_hash, source_doc, cleaned, raw_text=t)
                    self.current_core_value = prev_cv
                    self.current_sub_value = prev_sv
                    self.state = prev_state

                self._subvalues_table_col = (self._subvalues_table_col + 1) % len(self._subvalues_table_core_order)
                return

            self._try_add_sub_value(para, doc_hash, source_doc, text)
        elif self.state == ExtractorState.IN_PILLAR:
            if self.current_pillar and not self.current_pillar.description_ar:
                if len(text) > 20:
                    self.current_pillar.description_ar = text
        elif self.state in (ExtractorState.IN_CORE_VALUE, ExtractorState.IN_SUB_VALUE):
            self._maybe_assign_fallback_definition(para, doc_hash, source_doc, text)
            if (
                self.state == ExtractorState.IN_CORE_VALUE
                and self.current_core_value
                and not self.current_definition_paras
                and not self.current_evidence_paras
                and self._looks_like_list_item(text)
                and len(text.strip()) <= 80
            ):
                self._try_add_sub_value(para, doc_hash, source_doc, text)
                return

