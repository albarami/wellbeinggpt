"""Sanity & Integrity Validator for evidence chunks.

Reason: A scholar is not just a generator; he is a critic of his own sources.
This module detects internal inconsistencies and known-pattern violations
in the corpus data, preventing propagation of malformed evidence.

Key checks:
1. Mixed attribution detection (Quran verse + hadith ref in same line)
2. Source type validation (quran chunks shouldn't contain hadith markers)
3. Quarantine flagging for problematic chunks
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Any


class IntegrityIssue(str, Enum):
    """Types of integrity issues that can be detected."""
    
    MIXED_ATTRIBUTION = "MIXED_ATTRIBUTION"  # Quran + hadith in same line
    INVALID_SOURCE_TYPE = "INVALID_SOURCE_TYPE"  # source_type doesn't match content
    MISSING_SOURCE_TYPE = "MISSING_SOURCE_TYPE"  # No source_type metadata
    MALFORMED_REFERENCE = "MALFORMED_REFERENCE"  # Reference format is wrong


@dataclass
class IntegrityResult:
    """Result of integrity validation for a chunk or span."""
    
    chunk_id: str
    is_valid: bool
    issues: list[IntegrityIssue]
    details: list[str]
    quarantined: bool = False
    suggested_split: list[dict[str, Any]] | None = None


# Quran verse patterns (Arabic)
QURAN_PATTERNS = [
    # Common Quran phrases that indicate a verse
    r"وأعدوا\s+لهم\s+ما\s+استطعتم",  # Al-Anfal 8:60
    r"إن\s+الله\s+يأمر\s+بالعدل",  # An-Nahl 16:90
    r"يا\s+أيها\s+الذين\s+آمنوا",  # Common opening
    r"قل\s+هو\s+الله\s+أحد",  # Al-Ikhlas
    r"الحمد\s+لله\s+رب\s+العالمين",  # Al-Fatiha
    r"بسم\s+الله\s+الرحمن\s+الرحيم",  # Basmala
    # Quranic citation markers
    r"\[[\u0600-\u06FF]+:\d+\]",  # [السورة:الآية]
    r"سورة\s+[\u0600-\u06FF]+",  # سورة الأنفال
    r"الآية\s+\d+",  # الآية 60
]

# Hadith collection markers
HADITH_MARKERS = [
    r"صحيح\s+مسلم",
    r"صحيح\s+البخاري",
    r"سنن\s+أبي\s+داود",
    r"سنن\s+الترمذي",
    r"سنن\s+النسائي",
    r"سنن\s+ابن\s+ماجه",
    r"مسند\s+أحمد",
    r"موطأ\s+مالك",
    r"شرح\s+صحيح\s+مسلم",
    r"فتح\s+الباري",
    r"\(مسلم[:\s]\d+\)",
    r"\(البخاري[:\s]\d+\)",
    r"رواه\s+مسلم",
    r"رواه\s+البخاري",
    r"أخرجه\s+مسلم",
    r"أخرجه\s+البخاري",
]

# Known problematic chunks (quarantine list)
QUARANTINED_CHUNKS = {
    "CH_268ba301e082": {
        "reason": "Mixed Quran/hadith attribution: Al-Anfal 8:60 cited as Sahih Muslim 1917",
        "suggested_fix": "Split into Quran verse (Al-Anfal 8:60) and hadith commentary (Muslim 1917)",
    },
}


def _has_quran_pattern(text: str) -> bool:
    """Check if text contains Quran verse patterns."""
    for pattern in QURAN_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            return True
    return False


def _has_hadith_marker(text: str) -> bool:
    """Check if text contains hadith collection markers."""
    for pattern in HADITH_MARKERS:
        if re.search(pattern, text, re.IGNORECASE):
            return True
    return False


def _extract_quran_part(text: str) -> str | None:
    """Extract the Quran verse portion from mixed text."""
    # Common pattern: verse text before hadith marker
    for pattern in HADITH_MARKERS:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            before = text[:match.start()].strip()
            if before and _has_quran_pattern(before):
                # Clean up trailing punctuation
                before = re.sub(r'[.،,\s]+$', '', before)
                return before
    return None


def _extract_hadith_part(text: str) -> str | None:
    """Extract the hadith portion from mixed text."""
    # Find where hadith commentary starts (usually after verse)
    # Look for phrases like "ألا إن" or hadith-specific content
    hadith_start_patterns = [
        r"ألا\s+إن",  # "Indeed, verily" - common hadith explanation opener
        r"قال\s+رسول\s+الله",
        r"عن\s+النبي",
    ]
    
    for pattern in hadith_start_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            # Include from this point to the end
            hadith_text = text[match.start():].strip()
            return hadith_text
    
    return None


def detect_mixed_attribution(text: str) -> tuple[bool, str | None]:
    """
    Detect if text has mixed Quran/hadith attribution.
    
    Returns:
        (is_mixed, detail_message)
    """
    has_quran = _has_quran_pattern(text)
    has_hadith = _has_hadith_marker(text)
    
    if has_quran and has_hadith:
        return True, "Text contains both Quran verse pattern and hadith reference marker"
    
    return False, None


def validate_chunk(chunk: dict[str, Any]) -> IntegrityResult:
    """
    Validate a single chunk for integrity issues.
    
    Args:
        chunk: Dict with keys like chunk_id, text_ar, source_type, chunk_type
        
    Returns:
        IntegrityResult with validation status and details
    """
    chunk_id = str(chunk.get("chunk_id") or "")
    text_ar = str(chunk.get("text_ar") or "")
    source_type = str(chunk.get("source_type") or "")
    
    issues: list[IntegrityIssue] = []
    details: list[str] = []
    suggested_split: list[dict[str, Any]] | None = None
    
    # Check if in quarantine list
    if chunk_id in QUARANTINED_CHUNKS:
        quarantine_info = QUARANTINED_CHUNKS[chunk_id]
        return IntegrityResult(
            chunk_id=chunk_id,
            is_valid=False,
            issues=[IntegrityIssue.MIXED_ATTRIBUTION],
            details=[quarantine_info["reason"]],
            quarantined=True,
            suggested_split=None,
        )
    
    # Check for mixed attribution
    is_mixed, mixed_detail = detect_mixed_attribution(text_ar)
    if is_mixed:
        issues.append(IntegrityIssue.MIXED_ATTRIBUTION)
        details.append(mixed_detail or "Mixed attribution detected")
        
        # Suggest split
        quran_part = _extract_quran_part(text_ar)
        hadith_part = _extract_hadith_part(text_ar)
        
        if quran_part or hadith_part:
            suggested_split = []
            if quran_part:
                suggested_split.append({
                    "source_type": "quran",
                    "text_ar": quran_part,
                    "note": "Extracted Quran verse portion",
                })
            if hadith_part:
                suggested_split.append({
                    "source_type": "hadith",
                    "text_ar": hadith_part,
                    "note": "Extracted hadith commentary portion",
                })
    
    # Check source_type consistency
    if source_type == "quran" and _has_hadith_marker(text_ar):
        issues.append(IntegrityIssue.INVALID_SOURCE_TYPE)
        details.append("Chunk marked as 'quran' but contains hadith markers")
    
    if source_type == "hadith" and _has_quran_pattern(text_ar) and not _has_hadith_marker(text_ar):
        issues.append(IntegrityIssue.INVALID_SOURCE_TYPE)
        details.append("Chunk marked as 'hadith' but appears to be Quran verse")
    
    return IntegrityResult(
        chunk_id=chunk_id,
        is_valid=len(issues) == 0,
        issues=issues,
        details=details,
        quarantined=False,
        suggested_split=suggested_split,
    )


def validate_evidence_packets(packets: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[IntegrityResult]]:
    """
    Validate a list of evidence packets and filter out quarantined ones.
    
    Args:
        packets: List of evidence packet dicts
        
    Returns:
        (valid_packets, validation_results)
        - valid_packets: Packets that passed validation (not quarantined)
        - validation_results: Full validation results for all packets
    """
    valid_packets: list[dict[str, Any]] = []
    results: list[IntegrityResult] = []
    
    for packet in packets:
        result = validate_chunk(packet)
        results.append(result)
        
        # Only include non-quarantined packets
        if not result.quarantined:
            valid_packets.append(packet)
    
    return valid_packets, results


def get_integrity_warning_message(results: list[IntegrityResult]) -> str | None:
    """
    Generate a user-friendly warning message if any integrity issues were found.
    
    Returns:
        Arabic warning message, or None if no issues
    """
    quarantined = [r for r in results if r.quarantined]
    flagged = [r for r in results if not r.is_valid and not r.quarantined]
    
    if not quarantined and not flagged:
        return None
    
    parts: list[str] = []
    
    if quarantined:
        parts.append(f"تم استبعاد {len(quarantined)} مصدر(مصادر) بسبب تضارب في صياغة الدليل")
    
    if flagged:
        parts.append(f"وجدت {len(flagged)} ملاحظة(ملاحظات) على جودة البيانات")
    
    return " | ".join(parts)


# Convenience function for quick validation
def is_chunk_quarantined(chunk_id: str) -> bool:
    """Check if a chunk ID is in the quarantine list."""
    return chunk_id in QUARANTINED_CHUNKS


def get_quarantine_reason(chunk_id: str) -> str | None:
    """Get the reason a chunk was quarantined."""
    if chunk_id in QUARANTINED_CHUNKS:
        return QUARANTINED_CHUNKS[chunk_id].get("reason")
    return None
