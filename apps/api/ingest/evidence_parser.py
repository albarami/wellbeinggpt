"""
Evidence Parser Module

Parses Quran and Hadith references from Arabic text using regex patterns.
Implements dual storage (ref_raw + ref_norm) per the plan requirements.
"""

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class ParseStatus(str, Enum):
    """Status of reference parsing."""

    SUCCESS = "success"
    FAILED = "failed"
    NEEDS_REVIEW = "needs_review"


@dataclass
class ParsedQuranRef:
    """
    A parsed Quran reference.

    Attributes:
        surah_name_ar: Surah name in Arabic.
        surah_number: Surah number (1-114) if resolved.
        ayah_number: Ayah number.
        ayah_end: End ayah for range references.
        ref_raw: Original reference string as appears in document.
        ref_norm: Canonical normalized form for indexing.
        verse_text: The verse text if included.
        parse_status: Success/failed/needs_review.
    """

    surah_name_ar: str
    ayah_number: int
    ref_raw: str
    ref_norm: str
    surah_number: Optional[int] = None
    ayah_end: Optional[int] = None
    verse_text: str = ""
    parse_status: ParseStatus = ParseStatus.SUCCESS


@dataclass
class ParsedHadithRef:
    """
    A parsed Hadith reference.

    Attributes:
        collection: Hadith collection name (e.g., البخاري, مسلم).
        number: Hadith number.
        ref_raw: Original reference string.
        ref_norm: Canonical normalized form.
        hadith_text: The hadith text if included.
        parse_status: Success/failed/needs_review.
    """

    collection: str
    number: int
    ref_raw: str
    ref_norm: str
    hadith_text: str = ""
    parse_status: ParseStatus = ParseStatus.SUCCESS


@dataclass
class ParsedEvidence:
    """
    Container for parsed evidence from text.

    Attributes:
        quran_refs: List of parsed Quran references.
        hadith_refs: List of parsed Hadith references.
        unparsed_segments: Text segments that couldn't be parsed.
        raw_text: The original complete text.
    """

    quran_refs: list[ParsedQuranRef] = field(default_factory=list)
    hadith_refs: list[ParsedHadithRef] = field(default_factory=list)
    unparsed_segments: list[str] = field(default_factory=list)
    raw_text: str = ""


from apps.api.ingest.evidence_parser_data import (
    HADITH_COLLECTIONS,
    HADITH_PATTERN,
    HADITH_PATTERN_RAWAHU,
    QURAN_PATTERN_BARE,
    QURAN_PATTERN_BRACKET,
    QURAN_PATTERN_REF_ONLY,
    SURAH_NAMES,
)


class EvidenceParser:
    """
    Parser for Quran and Hadith references in Arabic text.

    This parser:
    1. Extracts references using regex patterns
    2. Normalizes references to canonical form
    3. Preserves original text (ref_raw)
    4. Marks parse failures without dropping evidence
    """

    def __init__(self):
        """Initialize the evidence parser."""
        pass

    def parse(self, text: str) -> ParsedEvidence:
        """
        Parse all evidence references from text.

        Args:
            text: Arabic text containing evidence references.

        Returns:
            ParsedEvidence: Parsed Quran and Hadith references.
        """
        result = ParsedEvidence(raw_text=text)

        # Parse Quran references
        result.quran_refs = self._parse_quran_refs(text)

        # Parse Hadith references
        result.hadith_refs = self._parse_hadith_refs(text)

        return result

    def _parse_quran_refs(self, text: str) -> list[ParsedQuranRef]:
        """Parse all Quran references from text."""
        refs = []

        # Try pattern with verse text first
        for match in QURAN_PATTERN_BRACKET.finditer(text):
            verse_text = match.group(1).strip()
            surah_name = match.group(2).strip()
            ayah_start = int(match.group(3))
            ayah_end = int(match.group(4)) if match.group(4) else None

            ref = self._create_quran_ref(
                surah_name, ayah_start, ayah_end, match.group(0), verse_text
            )
            refs.append(ref)

        # Try reference-only pattern
        for match in QURAN_PATTERN_REF_ONLY.finditer(text):
            # Skip if already captured by bracket pattern
            if any(match.group(0) in r.ref_raw for r in refs):
                continue

            surah_name = match.group(1).strip()
            ayah_start = int(match.group(2))
            ayah_end = int(match.group(3)) if match.group(3) else None

            ref = self._create_quran_ref(
                surah_name, ayah_start, ayah_end, match.group(0), ""
            )
            refs.append(ref)

        return refs

    def _create_quran_ref(
        self,
        surah_name: str,
        ayah_start: int,
        ayah_end: Optional[int],
        ref_raw: str,
        verse_text: str,
    ) -> ParsedQuranRef:
        """Create a ParsedQuranRef with normalization."""
        # Normalize surah name
        surah_info = self._normalize_surah_name(surah_name)

        if surah_info:
            canonical_name, surah_number = surah_info
            # Create normalized reference
            if ayah_end:
                ref_norm = f"{canonical_name}:{ayah_start}-{ayah_end}"
            else:
                ref_norm = f"{canonical_name}:{ayah_start}"

            return ParsedQuranRef(
                surah_name_ar=canonical_name,
                surah_number=surah_number,
                ayah_number=ayah_start,
                ayah_end=ayah_end,
                ref_raw=ref_raw,
                ref_norm=ref_norm,
                verse_text=verse_text,
                parse_status=ParseStatus.SUCCESS,
            )
        else:
            # Surah not recognized - mark for review
            if ayah_end:
                ref_norm = f"{surah_name}:{ayah_start}-{ayah_end}"
            else:
                ref_norm = f"{surah_name}:{ayah_start}"

            return ParsedQuranRef(
                surah_name_ar=surah_name,
                surah_number=None,
                ayah_number=ayah_start,
                ayah_end=ayah_end,
                ref_raw=ref_raw,
                ref_norm=ref_norm,
                verse_text=verse_text,
                parse_status=ParseStatus.NEEDS_REVIEW,
            )

    def _normalize_surah_name(
        self, name: str
    ) -> Optional[tuple[str, int]]:
        """
        Normalize a surah name to canonical form.

        Args:
            name: The surah name to normalize.

        Returns:
            Tuple of (canonical_name, surah_number) or None if not found.
        """
        # Clean up the name
        name = name.strip()

        # Remove "سورة" prefix if present
        name = re.sub(r"^سورة\s*", "", name)

        # Try direct lookup
        if name in SURAH_NAMES:
            return SURAH_NAMES[name]

        # Try without diacritics
        from apps.api.retrieve.normalize_ar import normalize_for_matching

        normalized = normalize_for_matching(name)
        for key, value in SURAH_NAMES.items():
            if normalize_for_matching(key) == normalized:
                return value

        return None

    def _parse_hadith_refs(self, text: str) -> list[ParsedHadithRef]:
        """Parse all Hadith references from text."""
        refs = []

        # Try standard pattern (collection: number)
        for match in HADITH_PATTERN.finditer(text):
            collection_raw = match.group(1).strip()
            number = int(match.group(2))

            ref = self._create_hadith_ref(
                collection_raw, number, match.group(0)
            )
            if ref:
                refs.append(ref)

        # Try "رواه" pattern
        for match in HADITH_PATTERN_RAWAHU.finditer(text):
            collection_raw = match.group(1).strip()
            number_str = match.group(2)
            number = int(number_str) if number_str else 0

            # Accept even if number is missing (0), mark as needs_review
            ref = self._create_hadith_ref(
                collection_raw, number, match.group(0)
            )
            if ref:
                refs.append(ref)

        return refs

    def _create_hadith_ref(
        self,
        collection_raw: str,
        number: int,
        ref_raw: str,
    ) -> Optional[ParsedHadithRef]:
        """Create a ParsedHadithRef with normalization."""
        # Normalize collection name
        canonical_collection = self._normalize_collection_name(collection_raw)

        if canonical_collection:
            ref_norm = f"{canonical_collection}:{number}"
            return ParsedHadithRef(
                collection=canonical_collection,
                number=number,
                ref_raw=ref_raw,
                ref_norm=ref_norm,
                parse_status=ParseStatus.SUCCESS if number > 0 else ParseStatus.NEEDS_REVIEW,
            )
        else:
            # Check if it looks like a hadith collection
            if self._looks_like_hadith_collection(collection_raw):
                ref_norm = f"{collection_raw}:{number}"
                return ParsedHadithRef(
                    collection=collection_raw,
                    number=number,
                    ref_raw=ref_raw,
                    ref_norm=ref_norm,
                    parse_status=ParseStatus.NEEDS_REVIEW,
                )

        return None

    def _normalize_collection_name(self, name: str) -> Optional[str]:
        """Normalize a hadith collection name."""
        name = name.strip()

        if name in HADITH_COLLECTIONS:
            return HADITH_COLLECTIONS[name]

        # Try without diacritics
        from apps.api.retrieve.normalize_ar import normalize_for_matching

        normalized = normalize_for_matching(name)
        for key, value in HADITH_COLLECTIONS.items():
            key_norm = normalize_for_matching(key)
            if key_norm == normalized:
                return value
            # Allow extra honorifics/phrases: match collection name as substring
            if key_norm and key_norm in normalized:
                return value

        return None

    def _looks_like_hadith_collection(self, name: str) -> bool:
        """Check if a name looks like it could be a hadith collection."""
        # Look for common patterns
        patterns = [
            r"الإمام",
            r"الامام",
            r"صحيح",
            r"سنن",
            r"مسند",
        ]
        for pattern in patterns:
            if re.search(pattern, name):
                return True
        return False


def parse_evidence_text(text: str) -> ParsedEvidence:
    """
    Convenience function to parse evidence from text.

    Args:
        text: Arabic text containing evidence references.

    Returns:
        ParsedEvidence: Parsed references.
    """
    parser = EvidenceParser()
    return parser.parse(text)

