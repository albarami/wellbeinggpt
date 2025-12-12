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
    re.compile(r"^الركيزة\s+(الروحية|العاطفية|العقلية|الجسدية|الاجتماعية)", re.UNICODE),
    re.compile(r"^الحياة\s+(الروحية|العاطفية|العقلية|الجسدية|الاجتماعية)\s+الطيبة", re.UNICODE),
]

# Core value markers
CORE_VALUES_LIST_MARKERS = [
    "القيم الكلية",
    "الأمهات",
    "القيم الأمهات",
]

# Sub-value markers
SUB_VALUES_LIST_MARKERS = [
    "القيم الجزئية",
    "الأحفاد",
    "القيم الأحفاد",
]

# Definition markers
DEFINITION_MARKERS = [
    "المفهوم:",
    "المفهوم :",
    "التعريف:",
    "التعريف :",
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
    source_anchor: dict
    para_index: int


@dataclass
class ExtractedDefinition:
    """An extracted definition block."""

    text_ar: str
    source_anchor: dict
    para_indices: list[int] = field(default_factory=list)


@dataclass
class ExtractedSubValue:
    """An extracted sub-value."""

    id: str
    name_ar: str
    source_anchor: dict
    para_index: int
    definition: Optional[ExtractedDefinition] = None
    evidence: list[ExtractedEvidence] = field(default_factory=list)


@dataclass
class ExtractedCoreValue:
    """An extracted core value."""

    id: str
    name_ar: str
    source_anchor: dict
    para_index: int
    definition: Optional[ExtractedDefinition] = None
    evidence: list[ExtractedEvidence] = field(default_factory=list)
    sub_values: list[ExtractedSubValue] = field(default_factory=list)


@dataclass
class ExtractedPillar:
    """An extracted pillar."""

    id: str
    name_ar: str
    source_anchor: dict
    para_index: int
    description_ar: Optional[str] = None
    core_values: list[ExtractedCoreValue] = field(default_factory=list)


@dataclass
class ExtractionResult:
    """Result of the extraction process."""

    source_doc_id: str
    source_file_hash: str
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
        self.current_definition_paras: list[tuple[str, int]] = []
        self.current_evidence_paras: list[tuple[str, int]] = []

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

        source_doc_id = f"DOC_{doc.file_hash[:16]}"

        # Process each paragraph through the state machine
        for para in doc.paragraphs:
            self._process_paragraph(para, doc.file_hash)

        # Finalize any open sections
        self._finalize_current_sections(doc.file_hash)

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
            source_file_hash=doc.file_hash,
            framework_version=self.framework_version,
            pillars=self.pillars,
            validation_errors=self.validation_errors,
            warnings=self.warnings,
            total_core_values=total_cv,
            total_sub_values=total_sv,
            total_evidence=total_ev,
        )

    def _process_paragraph(self, para: ParsedParagraph, doc_hash: str) -> None:
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
            self._handle_pillar_start(para, doc_hash, text)
            return

        if self._is_core_values_list_marker(normalized):
            self._handle_core_values_list()
            return

        if self._is_sub_values_list_marker(normalized):
            self._handle_sub_values_list()
            return

        if self._is_definition_marker(text):
            self._handle_definition_start(para, doc_hash, text)
            return

        if self._is_evidence_marker(text):
            self._handle_evidence_start(para, doc_hash, text)
            return

        # Continue current section
        self._continue_current_section(para, doc_hash, text)

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
        for marker in CORE_VALUES_LIST_MARKERS:
            if normalize_for_matching(marker) in normalized_text:
                return True
        return False

    def _is_sub_values_list_marker(self, normalized_text: str) -> bool:
        """Check if text marks sub-values list."""
        for marker in SUB_VALUES_LIST_MARKERS:
            if normalize_for_matching(marker) in normalized_text:
                return True
        return False

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
        self._finalize_current_sections(None)
        self.state = ExtractorState.IN_PILLARS_LIST

    def _handle_pillar_start(
        self, para: ParsedParagraph, doc_hash: str, text: str
    ) -> None:
        """Handle start of a new pillar."""
        self._finalize_current_sections(doc_hash)

        self.pillar_counter += 1
        pillar_id = f"P{self.pillar_counter:03d}"

        # Extract pillar name
        name = self._extract_pillar_name(text)

        self.current_pillar = ExtractedPillar(
            id=pillar_id,
            name_ar=name,
            source_anchor=self._make_anchor(para, doc_hash),
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
        self, para: ParsedParagraph, doc_hash: str, text: str
    ) -> None:
        """Handle start of a definition block."""
        self._finalize_definition_and_evidence()

        # Remove marker and get content
        content = text
        for marker in DEFINITION_MARKERS:
            content = content.replace(marker, "").strip()

        if content:
            self.current_definition_paras = [(content, para.para_index)]
        else:
            self.current_definition_paras = []

        self.state = ExtractorState.IN_DEFINITION

    def _handle_evidence_start(
        self, para: ParsedParagraph, doc_hash: str, text: str
    ) -> None:
        """Handle start of an evidence block."""
        # Finalize any pending definition first
        self._finalize_definition(doc_hash)

        # Remove marker and get content
        content = text
        for marker in EVIDENCE_MARKERS:
            content = content.replace(marker, "").strip()

        if content:
            self.current_evidence_paras = [(content, para.para_index)]
        else:
            self.current_evidence_paras = []

        self.state = ExtractorState.IN_EVIDENCE

    def _continue_current_section(
        self, para: ParsedParagraph, doc_hash: str, text: str
    ) -> None:
        """Continue accumulating content in current section."""
        if self.state == ExtractorState.IN_DEFINITION:
            self.current_definition_paras.append((text, para.para_index))

        elif self.state == ExtractorState.IN_EVIDENCE:
            self.current_evidence_paras.append((text, para.para_index))

        elif self.state == ExtractorState.IN_CORE_VALUES_LIST:
            # Each line might be a core value name
            self._try_add_core_value(para, doc_hash, text)

        elif self.state == ExtractorState.IN_SUB_VALUES_LIST:
            # Each line might be a sub-value name
            self._try_add_sub_value(para, doc_hash, text)

        elif self.state == ExtractorState.IN_PILLAR:
            # Could be description or other content
            if self.current_pillar and not self.current_pillar.description_ar:
                # First non-marker paragraph after pillar heading is description
                if len(text) > 20:  # Reasonable description length
                    self.current_pillar.description_ar = text

    def _try_add_core_value(
        self, para: ParsedParagraph, doc_hash: str, text: str
    ) -> None:
        """Try to add a core value from text."""
        # Clean up the text
        name = self._clean_value_name(text)

        if not name or len(name) < 2:
            return

        self.core_value_counter += 1
        cv_id = f"CV{self.core_value_counter:03d}"

        cv = ExtractedCoreValue(
            id=cv_id,
            name_ar=name,
            source_anchor=self._make_anchor(para, doc_hash),
            para_index=para.para_index,
        )

        if self.current_pillar:
            self.current_pillar.core_values.append(cv)
            self.current_core_value = cv

    def _try_add_sub_value(
        self, para: ParsedParagraph, doc_hash: str, text: str
    ) -> None:
        """Try to add a sub-value from text."""
        name = self._clean_value_name(text)

        if not name or len(name) < 2:
            return

        self.sub_value_counter += 1
        sv_id = f"SV{self.sub_value_counter:03d}"

        sv = ExtractedSubValue(
            id=sv_id,
            name_ar=name,
            source_anchor=self._make_anchor(para, doc_hash),
            para_index=para.para_index,
        )

        if self.current_core_value:
            self.current_core_value.sub_values.append(sv)
            self.current_sub_value = sv

    def _clean_value_name(self, text: str) -> str:
        """Clean up a value name extracted from text."""
        # Remove numbering (Arabic or Western)
        text = re.sub(r"^[\d\u0660-\u0669]+[.\-)\s]+", "", text)
        # Remove bullet points
        text = re.sub(r"^[•\-–—]\s*", "", text)
        # Remove extra whitespace
        text = " ".join(text.split())
        return text.strip(" :.،")

    def _finalize_current_sections(self, doc_hash: Optional[str]) -> None:
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

        text = " ".join(t for t, _ in self.current_definition_paras)
        para_indices = [i for _, i in self.current_definition_paras]

        definition = ExtractedDefinition(
            text_ar=text,
            source_anchor={},  # Would need doc_hash
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

        # Evidence will be parsed by evidence_parser module
        # For now, store raw text with paragraph indices
        # This will be passed to evidence_parser later

        self.current_evidence_paras = []

    def _make_anchor(self, para: ParsedParagraph, doc_hash: str) -> dict:
        """Create a source anchor dictionary."""
        return {
            "source_doc_id": f"DOC_{doc_hash[:16]}",
            "anchor_type": "docx_para",
            "anchor_id": para.anchor_id,
            "anchor_range": None,
        }

