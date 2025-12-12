"""
Rule-Based Extractor Module

Implements a deterministic state machine to extract:
- Pillars (ركائز)
- Core Values (القيم الكلية / الأمهات)
- Sub-Values (القيم الجزئية / الأحفاد)
- Definitions (المفهوم)
- Evidence blocks (التأصيل)

This is the core of the "zero-hallucination" ingestion: all extraction
is rule-based with no LLM involvement.
"""

import re
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional

from apps.api.ingest.docx_reader import ParsedDocument, ParsedParagraph
from apps.api.retrieve.normalize_ar import normalize_for_matching


# =============================================================================
# State Machine States
# =============================================================================


class ExtractorState(Enum):
    """States for the extraction state machine."""

    INITIAL = auto()
    IN_PILLARS_LIST = auto()
    IN_PILLAR = auto()
    IN_CORE_VALUES_LIST = auto()
    IN_CORE_VALUE = auto()
    IN_SUB_VALUES_LIST = auto()
    IN_SUB_VALUE = auto()
    IN_DEFINITION = auto()
    IN_EVIDENCE = auto()


# =============================================================================
# Marker Patterns
# =============================================================================

# Pillar markers
PILLARS_LIST_MARKERS = [
    "ركائز الحياة الطيبة",
    "الركائز الخمس",
]

PILLAR_MARKERS = [
    re.compile(r"^(أولا|ثانيا|ثالثا|رابعا|خامسا)[.\s:]*الركيزة", re.UNICODE),
    re.compile(r"^الركيزة\s+(الروحية|العاطفية|الفكرية|البدنية|الاجتماعية)", re.UNICODE),
    # Some documents use "الحياة <pillar> الطيبة" as a standalone heading line (must end there)
    re.compile(
        r"^(أولا|ثانيا|ثالثا|رابعا|خامسا)?[.\s:]*الحياة\s+(الروحية|العاطفية|الفكرية|البدنية|الاجتماعية)\s+الطيبة\s*[:：]?\s*$",
        re.UNICODE,
    ),
]

# Core value markers
CORE_VALUES_LIST_MARKERS = [
    "القيم الكلية",
    "القيم الأمهات",
]

# Sub-value markers
SUB_VALUES_LIST_MARKERS = [
    "القيم الجزئية",
    "القيم الأحفاد",
]

# Definition markers
DEFINITION_MARKERS = [
    "المفهوم",
    "المفهوم:",
    "المفهوم :",
    "التعريف",
    "التعريف:",
    "التعريف :",
    "التعريف الإجرائي",
    "التعريف الإجرائي:",
    "التعريف الإجرائي :",
]

# Evidence markers
EVIDENCE_MARKERS = [
    "التأصيل",
    "التأصيل:",
    "التأصيل :",
    "الأدلة",
    "الأدلة:",
    "الأدلة :",
    "الدليل",
    "الدليل:",
    "الدليل :",
    "الشواهد",
    "الشواهد:",
    "الشواهد :",
]

# Section end markers (any of these starts a new section)
SECTION_END_MARKERS = DEFINITION_MARKERS + EVIDENCE_MARKERS + [
    "القيم الكلية",
    "القيم الجزئية",
]


# =============================================================================
# Extracted Data Structures
# =============================================================================


@dataclass
class ExtractedEvidence:
    """An extracted evidence reference."""

    evidence_type: str  # quran | hadith | book
    ref_raw: str
    text_ar: str
    source_doc: str
    source_hash: str
    source_anchor: str
    raw_text: str
    para_index: int


@dataclass
class ExtractedDefinition:
    """An extracted definition block."""

    text_ar: str
    source_doc: str
    source_hash: str
    source_anchor: str
    raw_text: str
    para_indices: list[int] = field(default_factory=list)


@dataclass
class ExtractedSubValue:
    """An extracted sub-value."""

    id: str
    name_ar: str
    source_doc: str
    source_hash: str
    source_anchor: str
    raw_text: str
    para_index: int
    definition: Optional[ExtractedDefinition] = None
    evidence: list[ExtractedEvidence] = field(default_factory=list)


@dataclass
class ExtractedCoreValue:
    """An extracted core value."""

    id: str
    name_ar: str
    source_doc: str
    source_hash: str
    source_anchor: str
    raw_text: str
    para_index: int
    definition: Optional[ExtractedDefinition] = None
    evidence: list[ExtractedEvidence] = field(default_factory=list)
    sub_values: list[ExtractedSubValue] = field(default_factory=list)


@dataclass
class ExtractedPillar:
    """An extracted pillar."""

    id: str
    name_ar: str
    source_doc: str
    source_hash: str
    source_anchor: str
    raw_text: str
    para_index: int
    description_ar: Optional[str] = None
    core_values: list[ExtractedCoreValue] = field(default_factory=list)


@dataclass
class ExtractionResult:
    """Result of the extraction process."""

    source_doc_id: str
    source_file_hash: str
    source_doc: str
    framework_version: str
    pillars: list[ExtractedPillar] = field(default_factory=list)
    validation_errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    total_core_values: int = 0
    total_sub_values: int = 0
    total_evidence: int = 0


# =============================================================================
# Rule-Based Extractor
# =============================================================================


class RuleExtractor:
    """
    Deterministic rule-based extractor for Arabic wellbeing framework.

    This implements a state machine that processes paragraphs in order,
    detecting markers and extracting structured content.
    """

    def __init__(self, framework_version: str = "2025-10"):
        """
        Initialize the extractor.

        Args:
            framework_version: Version string for the framework being extracted.
        """
        self.framework_version = framework_version
        self._reset()

    def _reset(self):
        """Reset internal state for a new extraction."""
        self.state = ExtractorState.INITIAL
        self.current_pillar: Optional[ExtractedPillar] = None
        self.current_core_value: Optional[ExtractedCoreValue] = None
        self.current_sub_value: Optional[ExtractedSubValue] = None
        # Accumulators store both raw_text and cleaned_text (for definition/evidence markers)
        self.current_definition_paras: list[dict] = []  # {"raw": str, "clean": str, "i": int, "a": str}
        self.current_evidence_paras: list[dict] = []  # {"raw": str, "clean": str, "i": int, "a": str}

        # Counters for ID generation
        self.pillar_counter = 0
        self.core_value_counter = 0
        self.sub_value_counter = 0

        # Results
        self.pillars: list[ExtractedPillar] = []
        self.validation_errors: list[str] = []
        self.warnings: list[str] = []

        # Authoritative skeleton extracted from the "ركائز الحياة الطيبة" table:
        # canonical pillar name -> list of (expected) core values (usually 3).
        self._skeleton_core_values_by_pillar: dict[str, list[str]] = {}
        # Paragraph indices that belong to the skeleton table region; skip in main state machine.
        self._skip_para_indices: set[int] = set()
        # If skeleton detected, we pre-create the 5 pillars here (stable ids) and reuse them.
        self._pillars_by_canonical: dict[str, ExtractedPillar] = {}

        # Pillar-section sub-values table mode (3-column tables under each pillar)
        self._subvalues_table_mode: bool = False
        self._subvalues_table_core_order: list[ExtractedCoreValue] = []
        self._subvalues_table_col: int = 0

    def extract(self, doc: ParsedDocument) -> ExtractionResult:
        """
        Extract structured content from a parsed document.

        Args:
            doc: The parsed DOCX document.

        Returns:
            ExtractionResult: The extracted pillars, values, and evidence.
        """
        self._reset()

        source_doc_id = f"DOC_{doc.doc_hash[:16]}"
        source_doc = f"docs/source/{doc.doc_name}"

        # Pre-scan for the framework skeleton table ("ركائز الحياة الطيبة") and mark
        # its paragraphs as skippable so we don't create duplicate pillars/core-values
        # from table headers.
        self._extract_framework_skeleton(doc)

        # Process each paragraph through the state machine
        for para in doc.paragraphs:
            if para.para_index in self._skip_para_indices:
                continue
            self._process_paragraph(para, doc.doc_hash, source_doc)

        # Finalize any open sections
        self._finalize_current_sections(doc.doc_hash, source_doc)

        # Calculate totals
        total_cv = sum(len(p.core_values) for p in self.pillars)
        total_sv = sum(
            len(cv.sub_values)
            for p in self.pillars
            for cv in p.core_values
        )
        total_ev = sum(
            len(cv.evidence) + sum(len(sv.evidence) for sv in cv.sub_values)
            for p in self.pillars
            for cv in p.core_values
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

    def _process_paragraph(self, para: ParsedParagraph, doc_hash: str, source_doc: str) -> None:
        """Process a single paragraph through the state machine."""
        text = para.text.strip()

        # In pillar sub-values table mode, empty cells must be counted to keep column alignment.
        if not text:
            if self.state == ExtractorState.IN_SUB_VALUES_LIST and self._subvalues_table_mode:
                self._subvalues_table_col = (self._subvalues_table_col + 1) % max(1, len(self._subvalues_table_core_order) or 3)
            return

        normalized = normalize_for_matching(text)

        # Check for state transitions based on markers
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

        # After pillar/list transitions, detect core/sub headings (context-sensitive)
        # If a skeleton table exists, core value headings may appear as plain names (without ordinals),
        # especially inside tables. In that case, accept only if the name matches the allowed
        # 3 core values for the current pillar.
        if self.current_pillar and self._skeleton_core_values_by_pillar:
            candidate = text.strip().rstrip(":：").strip()
            # Guardrails: avoid matching long prose that contains core-value keywords.
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

        # Detect core value headings that don't use "القيم الكلية" lists (common in this DOCX)
        if self._is_core_value_heading(text) and self.current_pillar:
            cleaned = self._clean_core_value_heading_name(text)
            if self._core_value_allowed_for_current_pillar(cleaned):
                self._finalize_definition_and_evidence()
                self._subvalues_table_mode = False
                self._start_core_value_from_heading(para, doc_hash, source_doc, text)
                return

        # Detect sub-value headings inside a core value (e.g., "التوحيد:" / "العزة")
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

        # Continue current section
        self._continue_current_section(para, doc_hash, source_doc, text)

    def _is_core_value_heading(self, text: str) -> bool:
        """
        Heuristic detection for core value headings in the framework.

        Examples:
        - "أولًا: قيمة الإيمان"
        - "ثانيا: قيمة العبادة"
        - "أولاً: التوازن العاطفي   Emotional Balance"
        - "أولا. البحث عن الحقيقة"
        """
        t = text.strip()
        if len(t) > 120:
            return False
        # Must begin with ordinal/numbering pattern
        if not re.match(r"^(أولا|أولًا|أولاً|ثانيا|ثانيًا|ثالثا|ثالثًا|رابعا|خامسا|سادسا|سابعا|ثامنا|تاسعا|عاشرا|[0-9\u0660-\u0669]+)[.\s:：\-]+", t):
            return False
        # Exclude pillar headings already handled
        if self._is_pillar_marker(t):
            return False
        # Exclude anything that looks like a "life pillar" heading even if regex misses
        n = normalize_for_matching(t)
        if normalize_for_matching("الحياة") in n and normalize_for_matching("الطيبة") in n:
            return False
        # Exclude list headings ("القيم ...")
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
        """
        If skeleton exists, map variant headings to the canonical core value name.
        Example: "العزيمة والصمود" -> "العزيمة" (as listed in skeleton table).
        """
        if not self._skeleton_core_values_by_pillar:
            return candidate
        p = self._canonicalize_pillar_name(pillar_name)
        allowed = self._skeleton_core_values_by_pillar.get(p) or []
        cand_n = normalize_for_matching(candidate)
        for a in allowed:
            a_n = normalize_for_matching(a)
            if a_n == cand_n:
                return a
        # allow substring mapping only for short headings
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
        # If we have a skeleton table, only allow core values listed for the current pillar.
        if not self.current_pillar:
            return False
        if not self._skeleton_core_values_by_pillar:
            return True
        pillar_key = self._canonicalize_pillar_name(self.current_pillar.name_ar)
        allowed = self._skeleton_core_values_by_pillar.get(pillar_key) or []
        n = normalize_for_matching(core_value_name)
        if not n:
            return False

        # Safety: only permit fuzzy/substring matching for short, heading-like strings.
        looks_like_heading = (len(core_value_name) <= 40) and (len(core_value_name.split()) <= 4)

        for a in allowed:
            a_n = normalize_for_matching(a)
            if a_n == n:
                return True
            if looks_like_heading and a_n and (a_n in n):
                # e.g., "العزيمة" within "العزيمة والصمود"
                return True
        return False

    def _start_core_value_from_heading(self, para: ParsedParagraph, doc_hash: str, source_doc: str, text: str) -> None:
        name = text.strip()
        # Strip ordinal and the word "قيمة" if present
        name = re.sub(r"^(أولا|أولًا|أولاً|ثانيا|ثانيًا|ثالثا|ثالثًا|رابعا|خامسا|سادسا|سابعا|ثامنا|تاسعا|عاشرا|[0-9\u0660-\u0669]+)[.\s:：\-]+", "", name).strip()
        name = re.sub(r"^\s*قيمة\s+", "", name).strip()
        # Remove English tail if present after multiple spaces
        if "  " in name:
            name = name.split("  ")[0].strip()

        if self.current_pillar and self._skeleton_core_values_by_pillar:
            name = self._canonicalize_core_value_name_for_pillar(self.current_pillar.name_ar, name if name else text.strip())

        # De-dupe within the current pillar by name (prevents double-counting from repeated headings)
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
        """
        Heuristic detection for sub-value headings inside a core value.

        Examples:
        - "التوحيد:"
        - "العلم عن الله:"
        - "الإخلاص"
        - "التغذية السليمة"
        """
        t = text.strip()
        if not t or len(t) > 80:
            return False
        # Exclude section markers
        if self._is_definition_marker(t) or self._is_evidence_marker(t):
            return False
        # Exclude common headings that are not entities
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
        # Exclude verses/hadith lines
        if t.startswith("﴿") or t.startswith("{") or t.startswith("(") or t.startswith("«"):
            return False
        if t.startswith("قال") or t.startswith("وقال") or t.startswith("يقول") or t.startswith("وفي الحديث"):
            return False
        # Exclude evidence-intro prose that often appears inline
        if "يقول الله تعالى" in t or "قال الله تعالى" in t or t.startswith("وفي"):
            return False
        # Headings often end with ":" or are short standalone lines
        if t.endswith(":") or t.endswith("："):
            return True
        return False

    def _start_sub_value_from_heading(self, para: ParsedParagraph, doc_hash: str, source_doc: str, text: str) -> None:
        if not self.current_core_value:
            return
        name = text.strip().rstrip(":：").strip()
        self._add_sub_value_name(para, doc_hash, source_doc, name, raw_text=text)

    def _is_pillars_list_marker(self, normalized_text: str) -> bool:
        """Check if text marks the start of pillars list."""
        for marker in PILLARS_LIST_MARKERS:
            if normalize_for_matching(marker) in normalized_text:
                return True
        return False

    def _is_pillar_marker(self, text: str) -> bool:
        """Check if text marks a pillar heading."""
        for pattern in PILLAR_MARKERS:
            if pattern.search(text):
                return True
        return False

    def _is_core_values_list_marker(self, normalized_text: str) -> bool:
        """Check if text marks core values list."""
        # Be strict: must look like a heading starting with "القيم"
        if not normalized_text.startswith(normalize_for_matching("القيم")):
            return False
        return (
            normalize_for_matching("الكلية") in normalized_text
            or normalize_for_matching("الأمهات") in normalized_text
        )

    def _is_sub_values_list_marker(self, normalized_text: str) -> bool:
        """Check if text marks sub-values list."""
        if not normalized_text.startswith(normalize_for_matching("القيم")):
            return False
        return (
            normalize_for_matching("الجزئية") in normalized_text
            or normalize_for_matching("الأحفاد") in normalized_text
        )

    def _is_definition_marker(self, text: str) -> bool:
        """Check if text starts a definition block."""
        n = normalize_for_matching(text[:80])
        for marker in DEFINITION_MARKERS:
            if normalize_for_matching(marker) in n:
                return True
        return False

    def _is_evidence_marker(self, text: str) -> bool:
        """Check if text starts an evidence block."""
        n = normalize_for_matching(text[:80])
        for marker in EVIDENCE_MARKERS:
            if normalize_for_matching(marker) in n:
                return True
        return False

    def _handle_pillars_list(self) -> None:
        """Handle transition to pillars list state."""
        self._finalize_definition_and_evidence()
        self.state = ExtractorState.IN_PILLARS_LIST

    def _handle_pillar_start(
        self, para: ParsedParagraph, doc_hash: str, source_doc: str, text: str
    ) -> None:
        """Handle start of a new pillar."""
        # Finalize open definition/evidence when changing pillar context.
        self._finalize_definition_and_evidence()

        name = self._canonicalize_pillar_name(self._extract_pillar_name(text))

        # If skeleton exists, only allow the 5 canonical pillars from the table,
        # and reuse the pre-created pillar objects (stable IDs).
        if self._pillars_by_canonical:
            if name not in self._pillars_by_canonical:
                return
            self.current_pillar = self._pillars_by_canonical[name]
            # Prefer section heading anchor over table header anchor if available.
            if self.current_pillar.source_anchor.startswith("para_") and self.current_pillar.raw_text:
                # If current raw_text looks like table header ("... الطيبة") and new heading is richer, update.
                if ("الطيبة" in (self.current_pillar.raw_text or "")) and ("الركيزة" in text or "أولا" in text or "ثانيا" in text):
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

        # Legacy behavior (no skeleton): create new pillar
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
        """Extract the pillar name from heading text."""
        # Try to extract just the pillar name part
        # Remove ordinal prefixes
        name = re.sub(r"^(أولا|ثانيا|ثالثا|رابعا|خامسا)[.\s:]*", "", text)
        # Remove "الركيزة" prefix
        name = re.sub(r"^الركيزة\s*", "", name)
        # Clean up
        name = name.strip(" :.،")
        return name if name else text

    def _extract_framework_skeleton(self, doc: ParsedDocument) -> None:
        """
        Extract the authoritative framework skeleton from the "ركائز الحياة الطيبة" table:
        5 pillars x 3 core values.

        DocxReader flattens table cell paragraphs in row-major order, so we locate the marker then
        parse the next header row (5 cells) + 3 rows (15 cells).
        """
        marker_norm = normalize_for_matching("ركائز الحياة الطيبة")
        # Find marker paragraph index
        marker_pi: Optional[int] = None
        for p in doc.paragraphs:
            if marker_norm in normalize_for_matching(p.text or ""):
                marker_pi = p.para_index
                break
        if marker_pi is None:
            return

        # Build a window of subsequent non-empty paras (table cell texts) after marker
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
                normalize_for_matching(x) in n
                for x in ["الروحية", "العاطفية", "الفكرية", "البدنية", "الاجتماعية"]
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

        # Mark the marker paragraph + the header row + the 3 rows of core values as skippable.
        # (Skip by global para_index.)
        to_skip = [window[0]] + window[header_pos : header_pos + 5 + 15]
        self._skip_para_indices = {p.para_index for p in to_skip}

        # Pre-create the 5 canonical pillars with stable IDs (P001..P005) in the table order.
        # Use the table header cell anchor as initial provenance; we may later replace it with the
        # main section heading anchor when encountered.
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
        """Handle transition to core values list state."""
        self._finalize_definition_and_evidence()
        self.state = ExtractorState.IN_CORE_VALUES_LIST

    def _handle_sub_values_list(self) -> None:
        """Handle transition to sub-values list state."""
        self._finalize_definition_and_evidence()
        self.state = ExtractorState.IN_SUB_VALUES_LIST

        # If we are inside a pillar section with 3 core values, this is typically a 3-column table
        # where each subsequent row has one sub-value per core value.
        self._subvalues_table_mode = False
        self._subvalues_table_core_order = []
        self._subvalues_table_col = 0
        if self._skeleton_core_values_by_pillar and self.current_pillar and len(self.current_pillar.core_values) >= 3:
            allowed = self._skeleton_core_values_by_pillar.get(self.current_pillar.name_ar) or []
            if len(allowed) >= 3:
                # Order core values per the table order from the skeleton
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

    def _handle_definition_start(
        self, para: ParsedParagraph, doc_hash: str, source_doc: str, text: str
    ) -> None:
        """Handle start of a definition block."""
        self._finalize_definition_and_evidence()

        raw_line = text
        clean = text
        for marker in DEFINITION_MARKERS:
            clean = clean.replace(marker, "").strip()

        self.current_definition_paras = [{
            "raw": raw_line,
            "clean": clean,
            "i": para.para_index,
            "a": para.source_anchor,
            "doc": source_doc,
            "hash": doc_hash,
        }]

        self.state = ExtractorState.IN_DEFINITION

    def _handle_evidence_start(
        self, para: ParsedParagraph, doc_hash: str, source_doc: str, text: str
    ) -> None:
        """Handle start of an evidence block."""
        # Finalize any pending definition first
        self._finalize_definition(doc_hash)

        raw_line = text
        clean = text
        for marker in EVIDENCE_MARKERS:
            clean = clean.replace(marker, "").strip()

        self.current_evidence_paras = [{
            "raw": raw_line,
            "clean": clean,
            "i": para.para_index,
            "a": para.source_anchor,
            "doc": source_doc,
            "hash": doc_hash,
        }]

        self.state = ExtractorState.IN_EVIDENCE

    def _continue_current_section(
        self, para: ParsedParagraph, doc_hash: str, source_doc: str, text: str
    ) -> None:
        """Continue accumulating content in current section."""
        if self.state == ExtractorState.IN_DEFINITION:
            self.current_definition_paras.append({
                "raw": text,
                "clean": text,
                "i": para.para_index,
                "a": para.source_anchor,
                "doc": source_doc,
                "hash": doc_hash,
            })

        elif self.state == ExtractorState.IN_EVIDENCE:
            self.current_evidence_paras.append({
                "raw": text,
                "clean": text,
                "i": para.para_index,
                "a": para.source_anchor,
                "doc": source_doc,
                "hash": doc_hash,
            })

        elif self.state == ExtractorState.IN_CORE_VALUES_LIST:
            # Each line might be a core value name
            self._try_add_core_value(para, doc_hash, source_doc, text)

        elif self.state == ExtractorState.IN_SUB_VALUES_LIST:
            if self._subvalues_table_mode and self._subvalues_table_core_order:
                # Table cells arrive row-major; after each "القيم الجزئية" marker row-start,
                # we assign the next 3 cells (including empty placeholders handled earlier)
                # to the 3 core values in order.
                if self._is_sub_values_list_marker(normalize_for_matching(text)):
                    self._subvalues_table_col = 0
                    return

                # Filter out non-name lines
                t = text.strip()
                if (
                    t
                    and len(t) <= 80
                    and not self._is_definition_marker(t)
                    and not self._is_evidence_marker(t)
                    and not (t.startswith("﴿") or t.startswith("{") or t.startswith("(") or t.startswith("«"))
                ):
                    cv = self._subvalues_table_core_order[self._subvalues_table_col]
                    # Temporarily set current_core_value so existing helper attaches correctly
                    prev_cv = self.current_core_value
                    prev_sv = self.current_sub_value
                    prev_state = self.state
                    self.current_core_value = cv
                    self._add_sub_value_name(para, doc_hash, source_doc, t, raw_text=t)
                    self.current_core_value = prev_cv
                    # Table listing only: do not enter IN_SUB_VALUE state (no definitions/evidence here)
                    self.current_sub_value = prev_sv
                    self.state = prev_state

                self._subvalues_table_col = (self._subvalues_table_col + 1) % len(self._subvalues_table_core_order)
                return

            # Fallback: Each line might be a sub-value name
            self._try_add_sub_value(para, doc_hash, source_doc, text)

        elif self.state == ExtractorState.IN_PILLAR:
            # Could be description or other content
            if self.current_pillar and not self.current_pillar.description_ar:
                # First non-marker paragraph after pillar heading is description
                if len(text) > 20:  # Reasonable description length
                    self.current_pillar.description_ar = text

        elif self.state in (ExtractorState.IN_CORE_VALUE, ExtractorState.IN_SUB_VALUE):
            # Fallback: if definition marker isn't used, treat first substantive paragraph as definition
            self._maybe_assign_fallback_definition(para, doc_hash, source_doc, text)
            # Also, some core values list sub-values as bullet/numbered items without a "القيم الجزئية" marker.
            # In that case, treat short list items as sub-value names.
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

    def _try_add_core_value(
        self, para: ParsedParagraph, doc_hash: str, source_doc: str, text: str
    ) -> None:
        """Try to add a core value from text."""
        # Only accept list-like short lines
        if not self._looks_like_list_item(text):
            return

        # Clean up the text
        name = self._clean_value_name(text)

        if not name or len(name) < 2:
            return

        if self.current_pillar and self._skeleton_core_values_by_pillar:
            name = self._canonicalize_core_value_name_for_pillar(self.current_pillar.name_ar, name)

        # If skeleton exists, enforce allowed core values per pillar
        if not self._core_value_allowed_for_current_pillar(name):
            return

        # De-dupe within pillar by name
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

    def _try_add_sub_value(
        self, para: ParsedParagraph, doc_hash: str, source_doc: str, text: str
    ) -> None:
        """Try to add a sub-value from text."""
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
        """
        Add a sub-value to the current core value from a plain name (no bullet/number required).
        Used for table cells under 'القيم الجزئية'.
        """
        if not self.current_core_value:
            return

        clean_name = (name or "").strip().rstrip(":：").strip()
        if not clean_name or len(clean_name) < 2:
            return

        # Reject obvious non-entity prose / section lines
        if "\n" in clean_name:
            return
        # Drop citation footnotes / bibliographic lines
        if re.match(r"^\d+\s*[-–—]?\s*", clean_name):
            return
        if normalize_for_matching(clean_name).startswith(normalize_for_matching("تفسير")):
            return
        if self._is_definition_marker(clean_name) or self._is_evidence_marker(clean_name):
            return
        if clean_name.startswith("قال تعالى") or clean_name.startswith("يقول الله تعالى") or clean_name.startswith("وفي الحديث"):
            return
        if clean_name.startswith("وتندرج") or clean_name.startswith("تندرج") or "قيم أحفاد" in clean_name:
            return
        if clean_name.startswith("ومنها") or clean_name.startswith("ومن"):
            return

        # De-dupe within core value
        n = normalize_for_matching(clean_name)
        for existing in self.current_core_value.sub_values:
            if normalize_for_matching(existing.name_ar) == n:
                # Do not create duplicates; keep existing.
                return

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


    def _clean_value_name(self, text: str) -> str:
        """Clean up a value name extracted from text."""
        # Guard: avoid treating long prose as list items
        if len(text) > 120 and ":" in text:
            # Likely a definition paragraph, not a list item name
            return ""

        # Remove numbering (Arabic or Western)
        text = re.sub(r"^[\d\u0660-\u0669]+[.\-)\s]+", "", text)
        # Remove Arabic ordinal markers like "أولا:" "ثانيا:"
        text = re.sub(r"^(أولا|ثانيا|ثالثا|رابعا|خامسا|سادسا|سابعا|ثامنا|تاسعا|عاشرا)\s*[:.\-)\s]+", "", text)
        # Remove bullet points
        text = re.sub(r"^[•\-–—]\s*", "", text)
        # If there is English label after Arabic, keep Arabic part (common in headings)
        if "  " in text:
            text = text.split("  ")[0]
        # Remove extra whitespace
        text = " ".join(text.split())
        return text.strip(" :.،")

    def _looks_like_list_item(self, text: str) -> bool:
        """
        Heuristic: determine if a paragraph is a list item line (value name),
        not a long explanatory paragraph.
        """
        t = text.strip()
        if not t:
            return False
        # Most value-name lines are relatively short
        if len(t) > 120:
            return False
        # Starts with bullet/number/ordinal, or contains a short "name:" pattern
        if re.match(r"^[•\-–—]\s*", t):
            return True
        if re.match(r"^[\d\u0660-\u0669]+[.\-)\s]+", t):
            return True
        if re.match(r"^(أولا|ثانيا|ثالثا|رابعا|خامسا|سادسا|سابعا|ثامنا|تاسعا|عاشرا)\s*[:.\-)\s]+", t):
            return True
        # Some documents list values as "الاسم:" with short lead-in
        if ":" in t and len(t.split(":")[0]) <= 40:
            return True
        return False

    def _maybe_assign_fallback_definition(
        self, para: ParsedParagraph, doc_hash: str, source_doc: str, text: str
    ) -> None:
        """
        Some sections provide a definition immediately after the value heading without 'المفهوم:'.
        This assigns the first substantive paragraph as the definition when missing.
        """
        t = text.strip()
        if not t:
            return
        # Avoid capturing list headings/markers
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
        """Finalize any open sections."""
        self._finalize_definition_and_evidence()

        # Add current pillar to list if exists (legacy mode).
        # In skeleton mode, pillars are pre-created and should not be appended again.
        if self.current_pillar and self.current_pillar not in self.pillars:
            self.pillars.append(self.current_pillar)

        self.current_pillar = None
        self.current_core_value = None
        self.current_sub_value = None

    def _finalize_definition_and_evidence(self) -> None:
        """Finalize pending definition and evidence."""
        self._finalize_definition(None)
        self._finalize_evidence()

    def _finalize_definition(self, doc_hash: Optional[str]) -> None:
        """Finalize the current definition block."""
        if not self.current_definition_paras:
            return

        text = " ".join(p["clean"] for p in self.current_definition_paras if p["clean"]).strip()
        raw_text = "\n".join(p["raw"] for p in self.current_definition_paras if p["raw"]).strip()
        para_indices = [p["i"] for p in self.current_definition_paras]
        first = self.current_definition_paras[0]

        definition = ExtractedDefinition(
            text_ar=text,
            source_doc=first.get("doc", ""),
            source_hash=first.get("hash", ""),
            source_anchor=first.get("a", ""),
            raw_text=raw_text,
            para_indices=para_indices,
        )

        # Assign to current entity
        if self.current_sub_value:
            self.current_sub_value.definition = definition
        elif self.current_core_value:
            self.current_core_value.definition = definition

        self.current_definition_paras = []

    def _finalize_evidence(self) -> None:
        """Finalize the current evidence block."""
        if not self.current_evidence_paras:
            return

        # Store as a raw evidence block on the current entity.
        # Parsing into Quran/Hadith refs happens in evidence_parser (pipeline step).
        raw_text = "\n".join(p["raw"] for p in self.current_evidence_paras if p["raw"]).strip()
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
        )

        if self.current_sub_value:
            self.current_sub_value.evidence.append(block)
        elif self.current_core_value:
            self.current_core_value.evidence.append(block)

        self.current_evidence_paras = []

