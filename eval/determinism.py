"""Determinism utilities.

All eval runs must be deterministic:
- fixed random seed
- deterministic sampling

Note: LLM outputs may still vary if the provider/model is non-deterministic.
For that reason, we log provider metadata and keep regression gates scoped.
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass


@dataclass(frozen=True)
class DeterminismConfig:
    seed: int = 1337


def set_global_determinism(cfg: DeterminismConfig) -> None:
    """Set global determinism knobs."""
    random.seed(cfg.seed)

    # Some libraries respect these env vars; set them deterministically.
    os.environ.setdefault("PYTHONHASHSEED", str(cfg.seed))
