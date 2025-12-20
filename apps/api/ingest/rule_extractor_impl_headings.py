"""Rule extractor implementation (heading detection / start blocks).

Reason: split `rule_extractor.py` into <500 LOC modules.
"""

from __future__ import annotations

import re

from apps.api.ingest.docx_reader import ParsedParagraph
from apps.api.retrieve.normalize_ar import normalize_for_matching

from apps.api.ingest.rule_extractor_types import ExtractedCoreValue, ExtractorState


class _RuleExtractorHeadingsMixin:
    def _is_core_value_heading(self, text: str) -> bool:
        t = text.strip()
        if len(t) > 120:
            return False
        if not re.match(
            r"^(أولا|أولًا|أولاً|ثانيا|ثانيًا|ثالثا|ثالثًا|رابعا|خامسا|سادسا|سابعا|ثامنا|تاسعا|عاشرا|[0-9\u0660-\u0669]+)[.\s:：\-]+",
            t,
        ):
            return False
        if self._is_pillar_marker(t):
            return False
        n = normalize_for_matching(t)
        if normalize_for_matching("الحياة") in n and normalize_for_matching("الطيبة") in n:
            return False
        if normalize_for_matching(t).startswith(normalize_for_matching("القيم")):
            return False
        return True

    def _clean_core_value_heading_name(self, text: str) -> str:
        name = text.strip()
        name = re.sub(
            r"^(أولا|أولًا|أولاً|ثانيا|ثانيًا|ثالثا|ثالثًا|رابعا|خامسا|سادسا|سابعا|ثامنا|تاسعا|عاشرا|[0-9\u0660-\u0669]+)[.\s:：\-]+",
            "",
            name,
        ).strip()
        name = re.sub(r"^\s*قيمة\s+", "", name).strip()
        if "  " in name:
            name = name.split("  ")[0].strip()
        return name

    def _canonicalize_core_value_name_for_pillar(self, pillar_name: str, candidate: str) -> str:
        if not self._skeleton_core_values_by_pillar:
            return candidate
        p = self._canonicalize_pillar_name(pillar_name)
        allowed = self._skeleton_core_values_by_pillar.get(p) or []
        cand_n = normalize_for_matching(candidate)
        for a in allowed:
            a_n = normalize_for_matching(a)
            if a_n == cand_n:
                return a
        if (len(candidate) <= 60) and (len(candidate.split()) <= 6):
            for a in allowed:
                a_n = normalize_for_matching(a)
                if a_n and a_n in cand_n:
                    return a
        return candidate

    def _canonicalize_pillar_name(self, text: str) -> str:
        t = normalize_for_matching(text)
        if normalize_for_matching("الروحية") in t:
            return "الحياة الروحية"
        if normalize_for_matching("العاطفية") in t:
            return "الحياة العاطفية"
        if normalize_for_matching("الفكرية") in t:
            return "الحياة الفكرية"
        if normalize_for_matching("البدنية") in t:
            return "الحياة البدنية"
        if normalize_for_matching("الاجتماعية") in t:
            return "الحياة الاجتماعية"
        return text.strip()

    def _core_value_allowed_for_current_pillar(self, core_value_name: str) -> bool:
        if not self.current_pillar:
            return False
        if not self._skeleton_core_values_by_pillar:
            return True
        pillar_key = self._canonicalize_pillar_name(self.current_pillar.name_ar)
        allowed = self._skeleton_core_values_by_pillar.get(pillar_key) or []
        n = normalize_for_matching(core_value_name)
        if not n:
            return False
        looks_like_heading = (len(core_value_name) <= 40) and (len(core_value_name.split()) <= 4)
        for a in allowed:
            a_n = normalize_for_matching(a)
            if a_n == n:
                return True
            if looks_like_heading and a_n and (a_n in n):
                return True
        return False

    def _start_core_value_from_heading(self, para: ParsedParagraph, doc_hash: str, source_doc: str, text: str) -> None:
        name = text.strip()
        name = re.sub(
            r"^(أولا|أولًا|أولاً|ثانيا|ثانيًا|ثالثا|ثالثًا|رابعا|خامسا|سادسا|سابعا|ثامنا|تاسعا|عاشرا|[0-9\u0660-\u0669]+)[.\s:：\-]+",
            "",
            name,
        ).strip()
        name = re.sub(r"^\s*قيمة\s+", "", name).strip()
        if "  " in name:
            name = name.split("  ")[0].strip()

        if self.current_pillar and self._skeleton_core_values_by_pillar:
            name = self._canonicalize_core_value_name_for_pillar(self.current_pillar.name_ar, name if name else text.strip())

        if self.current_pillar:
            target_norm = normalize_for_matching(name if name else text.strip())
            for existing in self.current_pillar.core_values:
                if normalize_for_matching(existing.name_ar) == target_norm:
                    self.current_core_value = existing
                    self.current_sub_value = None
                    self.state = ExtractorState.IN_CORE_VALUE
                    return

        self.core_value_counter += 1
        cv_id = f"CV{self.core_value_counter:03d}"
        cv = ExtractedCoreValue(
            id=cv_id,
            name_ar=name if name else text.strip(),
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

    def _is_sub_value_heading(self, text: str) -> bool:
        t = text.strip()
        if not t or len(t) > 80:
            return False
        if self._is_definition_marker(t) or self._is_evidence_marker(t):
            return False
        banned = {
            normalize_for_matching("المفهوم"),
            normalize_for_matching("المفهوم:"),
            normalize_for_matching("التأصيل"),
            normalize_for_matching("التأصيل:"),
            normalize_for_matching("الدليل"),
            normalize_for_matching("الدليل:"),
            normalize_for_matching("التعريف"),
            normalize_for_matching("التعريف:"),
        }
        if normalize_for_matching(t) in banned:
            return False
        if normalize_for_matching("لسان العرب") in normalize_for_matching(t):
            return False
        if normalize_for_matching("معجم") in normalize_for_matching(t) and normalize_for_matching("مادة") in normalize_for_matching(t):
            return False
        if t.startswith("﴿") or t.startswith("{") or t.startswith("«"):
            return False
        if t.startswith("(") and not re.match(r"^\\(\\s*[\\d\\u0660-\\u0669]{1,2}\\s*\\)", t):
            return False
        if t.startswith("قال") or t.startswith("وقال") or t.startswith("يقول") or t.startswith("وفي الحديث"):
            return False
        if "يقول الله تعالى" in t or "قال الله تعالى" in t or t.startswith("وفي"):
            return False
        if t.endswith(":") or t.endswith("："):
            return True
        # Note: place '-' first in character class to avoid range parsing issues on Windows/Python.
        if re.match(r"^[\\d\\u0660-\\u0669]{1,2}\\s*[-\\.\\)\\u060c:：]\\s+\\S+", t):
            return True
        if re.match(r"^\\(\\s*[\\d\\u0660-\\u0669]{1,2}\\s*\\)\\s+\\S+", t):
            return True
        if self._looks_like_list_item(t) and len(t.split()) <= 6 and ":" not in t:
            return True
        return False

    def _start_sub_value_from_heading(self, para: ParsedParagraph, doc_hash: str, source_doc: str, text: str) -> None:
        if not self.current_core_value:
            return
        name = self._clean_value_name(text.strip().rstrip(":：").strip())
        if not name:
            return
        n = normalize_for_matching(name)
        for existing in self.current_core_value.sub_values:
            if normalize_for_matching(existing.name_ar) == n:
                self.current_sub_value = existing
                self.state = ExtractorState.IN_SUB_VALUE
                return
        for existing in self.current_core_value.sub_values:
            ex_n = normalize_for_matching(existing.name_ar)
            if ex_n and (ex_n in n or n in ex_n):
                self.current_sub_value = existing
                self.state = ExtractorState.IN_SUB_VALUE
                return
        self._add_sub_value_name(para, doc_hash, source_doc, name, raw_text=text)

