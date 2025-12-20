"""Rule extractor implementation (building values + finalize blocks).

Reason: split `rule_extractor.py` into <500 LOC modules.
"""

from __future__ import annotations

import re
from typing import Optional

from apps.api.ingest.docx_reader import ParsedParagraph
from apps.api.retrieve.normalize_ar import normalize_for_matching

from apps.api.ingest.rule_extractor_types import (
    ExtractedCoreValue,
    ExtractedDefinition,
    ExtractedEvidence,
    ExtractedSubValue,
    ExtractorState,
)


class _RuleExtractorBuildFinalizeMixin:
    def _try_add_core_value(self, para: ParsedParagraph, doc_hash: str, source_doc: str, text: str) -> None:
        t0 = text.strip()
        if re.match(r"^\\d+\\s+.+", t0) and ("(" in t0 and ")" in t0):
            return
        if not self._looks_like_list_item(text):
            return

        name = self._clean_value_name(text)
        if not name or len(name) < 2:
            return

        if self.current_pillar and self._skeleton_core_values_by_pillar:
            name = self._canonicalize_core_value_name_for_pillar(self.current_pillar.name_ar, name)

        if not self._core_value_allowed_for_current_pillar(name):
            return

        if self.current_pillar:
            n = normalize_for_matching(name)
            for existing in self.current_pillar.core_values:
                if normalize_for_matching(existing.name_ar) == n:
                    self.current_core_value = existing
                    self.current_sub_value = None
                    self.state = ExtractorState.IN_CORE_VALUE
                    return

        self.core_value_counter += 1
        cv_id = f"CV{self.core_value_counter:03d}"
        cv = ExtractedCoreValue(
            id=cv_id,
            name_ar=name,
            source_doc=source_doc,
            source_hash=doc_hash,
            source_anchor=para.source_anchor,
            raw_text=text,
            para_index=para.para_index,
        )
        if self.current_pillar:
            self.current_pillar.core_values.append(cv)
            self.current_core_value = cv
            self.current_sub_value = None
            self.state = ExtractorState.IN_CORE_VALUE

    def _try_add_sub_value(self, para: ParsedParagraph, doc_hash: str, source_doc: str, text: str) -> None:
        t0 = text.strip()
        if re.match(r"^\\d+\\s+.+", t0) and ("(" in t0 and ")" in t0):
            return
        if not self._looks_like_list_item(text):
            return

        name = self._clean_value_name(text)
        if not name or len(name) < 2:
            return
        self._add_sub_value_name(para, doc_hash, source_doc, name, raw_text=text)

    def _add_sub_value_name(
        self,
        para: ParsedParagraph,
        doc_hash: str,
        source_doc: str,
        name: str,
        raw_text: Optional[str] = None,
    ) -> None:
        if not self.current_core_value:
            return

        clean_name = self._clean_value_name((name or "").strip()).strip().rstrip(":：").strip()
        if not clean_name or len(clean_name) < 2:
            return

        if "\\n" in clean_name:
            return
        if normalize_for_matching(clean_name) in {normalize_for_matching("التفصيل")}:
            return
        if re.search(r"(دار\\s+|بيروت|الطبعة|تحقيق|مكتبة|مجلة|جامعة|عام\\s+\\d{3,4})", clean_name):
            return
        if re.search(r"(المعجم\\s+|كشاف\\s+|مختار\\s+|التوقيف\\s+|دستور\\s+|اصطلاحات\\s+الفنون)", clean_name):
            return
        if re.search(r"\\(\\s*\\d+\\s*/\\s*\\d+\\s*\\)", clean_name):
            return
        if normalize_for_matching(clean_name).startswith(normalize_for_matching("تفسير")):
            return
        if self._is_definition_marker(clean_name) or self._is_evidence_marker(clean_name):
            return
        if clean_name.startswith("قال تعالى") or clean_name.startswith("يقول الله تعالى") or clean_name.startswith("وفي الحديث"):
            return
        if clean_name.startswith("و قال") or clean_name.startswith("وقال") or clean_name.startswith("قال"):
            return
        if clean_name.startswith("وتندرج") or clean_name.startswith("تندرج") or "قيم أحفاد" in clean_name:
            return
        if clean_name.startswith("ومنها") or clean_name.startswith("ومن"):
            return

        n = normalize_for_matching(clean_name)
        for existing in self.current_core_value.sub_values:
            if normalize_for_matching(existing.name_ar) == n:
                return

        if (
            self._skeleton_core_values_by_pillar
            and self.state != ExtractorState.IN_SUB_VALUES_LIST
            and self.current_pillar
            and normalize_for_matching(self.current_pillar.name_ar) == normalize_for_matching("الحياة الاجتماعية")
        ):
            allowed = self._allowed_subvalues_by_core.get(self.current_core_value.id)
            if allowed and n not in allowed and getattr(para, "style", "") != "OCR_USER_CONTEXT":
                return

        prev_state = self.state
        self.sub_value_counter += 1
        sv_id = f"SV{self.sub_value_counter:03d}"
        sv = ExtractedSubValue(
            id=sv_id,
            name_ar=clean_name,
            source_doc=source_doc,
            source_hash=doc_hash,
            source_anchor=para.source_anchor,
            raw_text=raw_text or clean_name,
            para_index=para.para_index,
        )
        self.current_core_value.sub_values.append(sv)
        self.current_sub_value = sv
        self.state = ExtractorState.IN_SUB_VALUE

        if (
            self._skeleton_core_values_by_pillar
            and self.current_pillar
            and normalize_for_matching(self.current_pillar.name_ar) == normalize_for_matching("الحياة الاجتماعية")
            and (prev_state == ExtractorState.IN_SUB_VALUES_LIST or getattr(para, "style", "") == "OCR_USER_CONTEXT")
        ):
            self._allowed_subvalues_by_core.setdefault(self.current_core_value.id, set()).add(n)

    def _clean_value_name(self, text: str) -> str:
        if len(text) > 120 and ":" in text:
            return ""
        text = re.sub(r"^\\(\\s*[\\d\\u0660-\\u0669]+\\s*\\)\\s*", "", text)
        # Note: place '-' first in character class to avoid range parsing issues on Windows/Python.
        text = re.sub(r"^[\\d\\u0660-\\u0669]+[-\\.\\)\\s]+", "", text)
        text = re.sub(
            r"^(أولا|ثانيا|ثالثا|رابعا|خامسا|سادسا|سابعا|ثامنا|تاسعا|عاشرا)\\s*[-:.\\)\\s]+",
            "",
            text,
        )
        text = re.sub(r"^[•\\-–—]\\s*", "", text)
        if "  " in text:
            text = text.split("  ")[0]
        text = " ".join(text.split())
        text = re.sub(r"\\s*[\\/／]\\s*", "/", text)
        text = text.replace("(في الأدوار الاجتماعية)", "في الأدوار الاجتماعية")
        text = text.replace("(قيمتان متلازمتان)", "(قيمة متلازمة)")
        if ":" in text:
            left, right = text.split(":", 1)
            if len(left.strip()) <= 40 and right.strip():
                text = left.strip()
        return text.strip(" :.،")

    def _looks_like_list_item(self, text: str) -> bool:
        t = text.strip()
        if not t:
            return False
        if len(t) > 120:
            return False
        if re.match(r"^[•\\-–—]\\s*", t):
            return True
        # Note: place '-' first in character class to avoid range parsing issues on Windows/Python.
        if re.match(r"^[\\d\\u0660-\\u0669]+[-\\.\\)\\s]+", t):
            return True
        if re.match(r"^(أولا|ثانيا|ثالثا|رابعا|خامسا|سادسا|سابعا|ثامنا|تاسعا|عاشرا)\\s*[-:.\\)\\s]+", t):
            return True
        if ":" in t and len(t.split(":")[0]) <= 40:
            left, right = t.split(":", 1)
            if not right.strip() or len(right.strip()) <= 5 or t.strip().endswith(":"):
                return True
            return False
        return False

    def _maybe_assign_fallback_definition(self, para: ParsedParagraph, doc_hash: str, source_doc: str, text: str) -> None:
        t = text.strip()
        if not t:
            return
        if self._is_core_values_list_marker(normalize_for_matching(t)):
            return
        if self._is_sub_values_list_marker(normalize_for_matching(t)):
            return
        if self._is_definition_marker(t) or self._is_evidence_marker(t):
            return
        if self._looks_like_list_item(t):
            return
        if len(t) < 40:
            return

        if self.current_sub_value and self.current_sub_value.definition is None:
            self.current_sub_value.definition = ExtractedDefinition(
                text_ar=t,
                source_doc=source_doc,
                source_hash=doc_hash,
                source_anchor=para.source_anchor,
                raw_text=t,
                para_indices=[para.para_index],
            )
            return

        if self.current_core_value and self.current_core_value.definition is None:
            self.current_core_value.definition = ExtractedDefinition(
                text_ar=t,
                source_doc=source_doc,
                source_hash=doc_hash,
                source_anchor=para.source_anchor,
                raw_text=t,
                para_indices=[para.para_index],
            )
            return

    def _finalize_current_sections(self, doc_hash: Optional[str], source_doc: Optional[str]) -> None:
        self._finalize_definition_and_evidence()
        if self.current_pillar and self.current_pillar not in self.pillars:
            self.pillars.append(self.current_pillar)
        self.current_pillar = None
        self.current_core_value = None
        self.current_sub_value = None

    def _finalize_definition_and_evidence(self) -> None:
        self._finalize_definition(None)
        self._finalize_evidence()

    def _finalize_definition(self, doc_hash: Optional[str]) -> None:
        if not self.current_definition_paras:
            return

        text = " ".join(p["clean"] for p in self.current_definition_paras if p["clean"]).strip()
        raw_text = "\\n".join(p["raw"] for p in self.current_definition_paras if p["raw"]).strip()
        para_indices = [p["i"] for p in self.current_definition_paras]
        first = self.current_definition_paras[0]

        definition = ExtractedDefinition(
            text_ar=text,
            source_doc=first.get("doc", ""),
            source_hash=first.get("hash", ""),
            source_anchor=first.get("a", ""),
            raw_text=raw_text,
            para_indices=para_indices,
            refs=list(self.current_block_refs),
        )

        if self.current_sub_value:
            self.current_sub_value.definition = definition
        elif self.current_core_value:
            self.current_core_value.definition = definition

        self.current_definition_paras = []
        self.current_block_refs = []

    def _finalize_evidence(self) -> None:
        if not self.current_evidence_paras:
            return

        raw_text = "\\n".join(p["raw"] for p in self.current_evidence_paras if p["raw"]).strip()
        text = " ".join(p["clean"] for p in self.current_evidence_paras if p["clean"]).strip()
        first = self.current_evidence_paras[0]

        block = ExtractedEvidence(
            evidence_type="evidence_block",
            ref_raw="",
            text_ar=text,
            source_doc=first.get("doc", ""),
            source_hash=first.get("hash", ""),
            source_anchor=first.get("a", ""),
            raw_text=raw_text,
            para_index=first.get("i", 0),
            refs=list(self.current_block_refs),
        )

        if self.current_sub_value:
            self.current_sub_value.evidence.append(block)
        elif self.current_core_value:
            self.current_core_value.evidence.append(block)

        self.current_evidence_paras = []
        self.current_block_refs = []

