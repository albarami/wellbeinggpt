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
    "المفهوم:",
    "المفهوم :",
    "التعريف:",
    "التعريف :",
    "التعريف الإجرائي:",
    "التعريف الإجرائي :",
]

# Evidence markers
EVIDENCE_MARKERS = [
    "التأصيل:",
    "التأصيل :",
    "الأدلة:",
    "الأدلة :",
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

        # Process each paragraph through the state machine
        for para in doc.paragraphs:
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

        if not text:
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

        if self._is_definition_marker(text):
            self._handle_definition_start(para, doc_hash, source_doc, text)
            return

        if self._is_evidence_marker(text):
            self._handle_evidence_start(para, doc_hash, source_doc, text)
            return

        # Continue current section
        self._continue_current_section(para, doc_hash, source_doc, text)

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
        for marker in DEFINITION_MARKERS:
            if text.startswith(marker) or marker in text[:50]:
                return True
        return False

    def _is_evidence_marker(self, text: str) -> bool:
        """Check if text starts an evidence block."""
        for marker in EVIDENCE_MARKERS:
            if text.startswith(marker) or marker in text[:50]:
                return True
        return False

    def _handle_pillars_list(self) -> None:
        """Handle transition to pillars list state."""
        self._finalize_current_sections(None, None)
        self.state = ExtractorState.IN_PILLARS_LIST

    def _handle_pillar_start(
        self, para: ParsedParagraph, doc_hash: str, source_doc: str, text: str
    ) -> None:
        """Handle start of a new pillar."""
        self._finalize_current_sections(doc_hash, source_doc)

        self.pillar_counter += 1
        pillar_id = f"P{self.pillar_counter:03d}"

        # Extract pillar name
        name = self._extract_pillar_name(text)

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

    def _handle_core_values_list(self) -> None:
        """Handle transition to core values list state."""
        self._finalize_definition_and_evidence()
        self.state = ExtractorState.IN_CORE_VALUES_LIST

    def _handle_sub_values_list(self) -> None:
        """Handle transition to sub-values list state."""
        self._finalize_definition_and_evidence()
        self.state = ExtractorState.IN_SUB_VALUES_LIST

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
            # Each line might be a sub-value name
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

        self.sub_value_counter += 1
        sv_id = f"SV{self.sub_value_counter:03d}"

        sv = ExtractedSubValue(
            id=sv_id,
            name_ar=name,
            source_doc=source_doc,
            source_hash=doc_hash,
            source_anchor=para.source_anchor,
            raw_text=text,
            para_index=para.para_index,
        )

        if self.current_core_value:
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

        # Add current pillar to list if exists
        if self.current_pillar:
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

