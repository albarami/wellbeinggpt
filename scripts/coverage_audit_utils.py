"""Coverage audit helpers.

Reason: keep each script file <500 LOC.
"""

from __future__ import annotations

import os
from typing import Any


def require_env(name: str) -> str:
    val = (os.getenv(name) or "").strip()
    if not val:
        raise RuntimeError(f"Missing required env var: {name}")
    return val


def anchor_str_from_canonical(anchor: Any) -> str:
    if isinstance(anchor, str):
        return anchor
    if isinstance(anchor, dict):
        if anchor.get("source_anchor"):
            return str(anchor.get("source_anchor") or "")
        if anchor.get("anchor_id"):
            return str(anchor.get("anchor_id") or "")
    return str(anchor or "")


def norm_ref_raw_for_match(s: str) -> str:
    if not s:
        return ""
    t = " ".join(str(s).split())
    t = t.strip().strip("()[]{}«»\"'،.؛:")
    return t

