"""
Arabic morphology helpers for entity resolution (non-LLM).

Goal: improve matching when users write inflected/attached forms:
- prefixes: و، ف، ب، ك، ل، ال...
- common suffixes/plurals: ون/ين/ات/ة...

This is intentionally conservative to avoid over-stemming and false positives.
"""

from __future__ import annotations

import re
from typing import Iterable

from apps.api.retrieve.normalize_ar import normalize_for_matching


_TOKEN_RE = re.compile(r"[^\w\u0600-\u06FF]+", re.UNICODE)

# Common single-letter prefixes that may attach to words (and/so/by/like/to).
_PREFIXES = ("و", "ف", "ب", "ك", "ل")

# Definite article variants.
_ARTICLE_PREFIXES = ("ال",)

# Common plural/derivational suffixes (conservative).
_SUFFIXES = (
    "ات",
    "ون",
    "ين",
    "ان",
    "ه",
    "ها",
    "هم",
    "هن",
    "كم",
    "كن",
    "نا",
    "ي",
    "ك",
    "ة",
)


def tokenize_ar(text: str) -> list[str]:
    """
    Tokenize Arabic-ish text into normalized tokens.

    Returns tokens in normalized form used by matching.
    """
    norm = normalize_for_matching(text)
    norm = _TOKEN_RE.sub(" ", norm)
    tokens = [t for t in norm.split() if t]
    return tokens


def generate_token_variants(token: str) -> set[str]:
    """
    Generate conservative variants for a single token.

    Strategy:
    - keep the token
    - strip a single clitic prefix (و/ف/ب/ك/ل) if present
    - strip definite article "ال" (optionally after clitic)
    - strip one suffix from a limited set
    """
    t = normalize_for_matching(token)
    variants: set[str] = {t}

    # Prefix stripping (one step)
    for p in _PREFIXES:
        if t.startswith(p) and len(t) >= 4:
            variants.add(t[len(p) :])

    # Article stripping (optionally after clitic)
    for v in list(variants):
        for a in _ARTICLE_PREFIXES:
            if v.startswith(a) and len(v) >= 4:
                variants.add(v[len(a) :])

        for p in _PREFIXES:
            if v.startswith(p + "ال") and len(v) >= 5:
                variants.add(v[len(p + "ال") :])

    # Suffix stripping (one step)
    for v in list(variants):
        for s in _SUFFIXES:
            if v.endswith(s) and len(v) - len(s) >= 3:
                variants.add(v[: -len(s)])

    # Avoid overly-short variants
    return {x for x in variants if len(x) >= 3}


def expand_query_terms(text: str) -> set[str]:
    """
    Expand a query into a set of tokens + variants.
    """
    out: set[str] = set()
    for tok in tokenize_ar(text):
        out |= generate_token_variants(tok)
    return out


def phrase_variants(phrase: str) -> set[str]:
    """
    Generate variants for a full phrase by token-variant expansion.

    Used to build a richer entity index.
    """
    tokens = tokenize_ar(phrase)
    if not tokens:
        return set()

    # Only generate phrase variants via token variants for the first 3 tokens
    # to keep combinatorics bounded.
    toks = tokens[:3]
    pools = [sorted(generate_token_variants(t)) for t in toks]

    variants: set[str] = set()
    # cartesian product (bounded by conservative variant counts)
    def _rec(i: int, acc: list[str]) -> None:
        if i == len(pools):
            variants.add(" ".join(acc))
            return
        for v in pools[i]:
            _rec(i + 1, acc + [v])

    _rec(0, [])
    variants.add(normalize_for_matching(phrase))
    return {v for v in variants if v}






