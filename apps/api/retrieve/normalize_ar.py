"""
Arabic Text Normalization Module

Provides deterministic normalization for Arabic text used in:
- Ingestion (consistent storage)
- Query normalization (consistent matching)
- Embedding (consistent vectors)
"""

import re
from typing import Optional


# =============================================================================
# Arabic Unicode Ranges and Characters
# =============================================================================

# Diacritics (Tashkeel) - optional vowel marks
ARABIC_DIACRITICS = re.compile(r"[\u064B-\u065F\u0670]")

# Tatweel (kashida) - elongation character
TATWEEL = "\u0640"

# Alef variants
ALEF_VARIANTS = {
    "\u0622": "\u0627",  # Alef with madda -> Alef
    "\u0623": "\u0627",  # Alef with hamza above -> Alef
    "\u0625": "\u0627",  # Alef with hamza below -> Alef
    "\u0671": "\u0627",  # Alef wasla -> Alef
}

# Hamza variants
HAMZA_VARIANTS = {
    "\u0624": "\u0621",  # Waw with hamza -> Hamza
    "\u0626": "\u0621",  # Yeh with hamza -> Hamza
}

# Yeh variants
YEH_VARIANTS = {
    "\u0649": "\u064A",  # Alef maksura -> Yeh
    "\u06CC": "\u064A",  # Farsi Yeh -> Arabic Yeh
}

# Teh marbuta to Heh (optional, context-dependent)
TEH_MARBUTA = "\u0629"
HEH = "\u0647"

# Arabic-Indic digits to Western digits
ARABIC_INDIC_DIGITS = {
    "\u0660": "0", "\u0661": "1", "\u0662": "2", "\u0663": "3", "\u0664": "4",
    "\u0665": "5", "\u0666": "6", "\u0667": "7", "\u0668": "8", "\u0669": "9",
    # Extended Arabic-Indic (Persian)
    "\u06F0": "0", "\u06F1": "1", "\u06F2": "2", "\u06F3": "3", "\u06F4": "4",
    "\u06F5": "5", "\u06F6": "6", "\u06F7": "7", "\u06F8": "8", "\u06F9": "9",
}

# Common Arabic punctuation
ARABIC_PUNCTUATION = {
    "\u060C": ",",  # Arabic comma
    "\u061B": ";",  # Arabic semicolon
    "\u061F": "?",  # Arabic question mark
    "\u06D4": ".",  # Arabic full stop
}


def remove_diacritics(text: str) -> str:
    """
    Remove Arabic diacritics (tashkeel) from text.

    This removes:
    - Fatha, Damma, Kasra
    - Shadda, Sukun
    - Tanween (Fathatan, Dammatan, Kasratan)
    - Superscript Alef

    Args:
        text: Arabic text with possible diacritics.

    Returns:
        Text with diacritics removed.
    """
    return ARABIC_DIACRITICS.sub("", text)


def remove_tatweel(text: str) -> str:
    """
    Remove tatweel (kashida) elongation characters.

    Args:
        text: Arabic text with possible tatweel.

    Returns:
        Text with tatweel removed.
    """
    return text.replace(TATWEEL, "")


def normalize_alef(text: str) -> str:
    """
    Normalize Alef variants to bare Alef.

    Converts:
    - Alef with madda above (آ) -> Alef (ا)
    - Alef with hamza above (أ) -> Alef (ا)
    - Alef with hamza below (إ) -> Alef (ا)
    - Alef wasla (ٱ) -> Alef (ا)

    Args:
        text: Arabic text.

    Returns:
        Text with normalized Alef.
    """
    for variant, normalized in ALEF_VARIANTS.items():
        text = text.replace(variant, normalized)
    return text


def normalize_hamza(text: str) -> str:
    """
    Normalize hamza-on-carrier variants to standalone hamza.

    Args:
        text: Arabic text.

    Returns:
        Text with normalized hamza.
    """
    for variant, normalized in HAMZA_VARIANTS.items():
        text = text.replace(variant, normalized)
    return text


def normalize_yeh(text: str) -> str:
    """
    Normalize Yeh variants to Arabic Yeh.

    Converts:
    - Alef maksura (ى) -> Yeh (ي)
    - Farsi Yeh (ی) -> Arabic Yeh (ي)

    Args:
        text: Arabic text.

    Returns:
        Text with normalized Yeh.
    """
    for variant, normalized in YEH_VARIANTS.items():
        text = text.replace(variant, normalized)
    return text


def normalize_teh_marbuta(text: str, to_heh: bool = False) -> str:
    """
    Optionally normalize Teh marbuta to Heh.

    This is context-dependent and should be used carefully.
    In some cases, preserving Teh marbuta is important for meaning.

    Args:
        text: Arabic text.
        to_heh: Whether to convert Teh marbuta to Heh.

    Returns:
        Text with optionally normalized Teh marbuta.
    """
    if to_heh:
        return text.replace(TEH_MARBUTA, HEH)
    return text


def normalize_digits(text: str) -> str:
    """
    Convert Arabic-Indic digits to Western (ASCII) digits.

    Args:
        text: Text with possible Arabic-Indic digits.

    Returns:
        Text with Western digits.
    """
    for arabic, western in ARABIC_INDIC_DIGITS.items():
        text = text.replace(arabic, western)
    return text


def normalize_punctuation(text: str) -> str:
    """
    Normalize Arabic punctuation to ASCII equivalents.

    Args:
        text: Text with Arabic punctuation.

    Returns:
        Text with normalized punctuation.
    """
    for arabic, ascii_char in ARABIC_PUNCTUATION.items():
        text = text.replace(arabic, ascii_char)
    return text


def normalize_whitespace(text: str) -> str:
    """
    Normalize whitespace: collapse multiple spaces, trim.

    Args:
        text: Text with irregular whitespace.

    Returns:
        Text with normalized whitespace.
    """
    return " ".join(text.split()).strip()


def normalize_arabic(
    text: str,
    remove_diacritics_flag: bool = True,
    normalize_alef_flag: bool = True,
    normalize_yeh_flag: bool = True,
    normalize_hamza_flag: bool = False,
    normalize_teh_marbuta_flag: bool = False,
    normalize_digits_flag: bool = True,
    normalize_punctuation_flag: bool = False,
    remove_tatweel_flag: bool = True,
) -> str:
    """
    Apply comprehensive Arabic normalization.

    This is the main normalization function that applies multiple
    normalization steps in the correct order.

    Args:
        text: Arabic text to normalize.
        remove_diacritics_flag: Remove diacritics (tashkeel).
        normalize_alef_flag: Normalize Alef variants.
        normalize_yeh_flag: Normalize Yeh variants.
        normalize_hamza_flag: Normalize hamza variants.
        normalize_teh_marbuta_flag: Convert Teh marbuta to Heh.
        normalize_digits_flag: Convert Arabic digits to Western.
        normalize_punctuation_flag: Normalize Arabic punctuation.
        remove_tatweel_flag: Remove tatweel.

    Returns:
        Normalized Arabic text.
    """
    if not text:
        return ""

    # Apply normalizations in order
    if remove_diacritics_flag:
        text = remove_diacritics(text)

    if remove_tatweel_flag:
        text = remove_tatweel(text)

    if normalize_alef_flag:
        text = normalize_alef(text)

    if normalize_hamza_flag:
        text = normalize_hamza(text)

    if normalize_yeh_flag:
        text = normalize_yeh(text)

    if normalize_teh_marbuta_flag:
        text = normalize_teh_marbuta(text, to_heh=True)

    if normalize_digits_flag:
        text = normalize_digits(text)

    if normalize_punctuation_flag:
        text = normalize_punctuation(text)

    # Always normalize whitespace
    text = normalize_whitespace(text)

    return text


def normalize_for_matching(text: str) -> str:
    """
    Normalize text for entity matching / exact lookup.

    This applies aggressive normalization suitable for matching
    entity names (pillars, values, etc.).

    Args:
        text: Text to normalize for matching.

    Returns:
        Normalized text suitable for exact matching.
    """
    return normalize_arabic(
        text,
        remove_diacritics_flag=True,
        normalize_alef_flag=True,
        normalize_yeh_flag=True,
        normalize_hamza_flag=True,
        normalize_teh_marbuta_flag=False,  # Keep for meaning
        normalize_digits_flag=True,
        normalize_punctuation_flag=True,
        remove_tatweel_flag=True,
    )


def normalize_for_embedding(text: str) -> str:
    """
    Normalize text for embedding / vector search.

    This applies moderate normalization suitable for semantic matching.

    Args:
        text: Text to normalize for embedding.

    Returns:
        Normalized text suitable for embedding.
    """
    return normalize_arabic(
        text,
        remove_diacritics_flag=True,
        normalize_alef_flag=True,
        normalize_yeh_flag=True,
        normalize_hamza_flag=False,
        normalize_teh_marbuta_flag=False,
        normalize_digits_flag=True,
        normalize_punctuation_flag=False,
        remove_tatweel_flag=True,
    )


def extract_arabic_words(text: str) -> list[str]:
    """
    Extract Arabic words from text, excluding stopwords.

    Used for claim-to-evidence checking.

    Args:
        text: Arabic text.

    Returns:
        List of Arabic words (normalized).
    """
    # Normalize first
    normalized = normalize_for_matching(text)

    # Extract Arabic letter sequences
    arabic_word_pattern = re.compile(r"[\u0600-\u06FF]+")
    words = arabic_word_pattern.findall(normalized)

    # Filter stopwords
    stopwords = get_arabic_stopwords()
    words = [w for w in words if w not in stopwords and len(w) > 1]

    return words


def get_arabic_stopwords() -> set[str]:
    """
    Get a set of common Arabic stopwords.

    These are common words that don't carry meaning for matching.

    Returns:
        Set of Arabic stopwords (normalized).
    """
    stopwords = {
        # Pronouns and particles
        "من", "الى", "على", "في", "عن", "مع", "هذا", "هذه", "ذلك", "تلك",
        "الذي", "التي", "الذين", "اللاتي", "اللواتي",
        # Conjunctions
        "و", "او", "ام", "ثم", "لكن", "بل", "حتى", "اذا", "اذ", "لو", "كي",
        # Prepositions
        "ب", "ك", "ل", "ف", "س",
        # Common verbs
        "كان", "يكون", "هو", "هي", "هم", "هن", "انا", "نحن", "انت", "انتم",
        # Articles and demonstratives
        "ال", "ان", "ما", "لا", "قد", "كل", "بعض", "غير",
        # Numbers
        "واحد", "اثنان", "ثلاثة", "اربعة", "خمسة",
    }

    # Normalize stopwords
    return {normalize_for_matching(w) for w in stopwords}


def detect_arabic_content(text: str) -> bool:
    """
    Check if text contains significant Arabic content.

    Args:
        text: Text to check.

    Returns:
        True if text contains Arabic letters.
    """
    arabic_pattern = re.compile(r"[\u0600-\u06FF]")
    matches = arabic_pattern.findall(text)
    return len(matches) > 0


def get_text_direction(text: str) -> str:
    """
    Determine text direction based on content.

    Args:
        text: Text to analyze.

    Returns:
        "rtl" for Arabic/right-to-left, "ltr" for left-to-right.
    """
    if detect_arabic_content(text):
        return "rtl"
    return "ltr"

