"""AlMuhasbi uplift scoring (A/B).

Compares FULL_SYSTEM vs RAG_PLUS_GRAPH.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable, Iterable


@dataclass(frozen=True)
class UpliftSummary:
    mean_delta: float
    ci_low: float
    ci_high: float


def bootstrap_ci(
    values: list[float],
    *,
    seed: int = 1337,
    iters: int = 500,
    alpha: float = 0.05,
) -> tuple[float, float]:
    if not values:
        return (0.0, 0.0)

    rnd = random.Random(seed)
    n = len(values)
    samples: list[float] = []
    for _ in range(iters):
        s = 0.0
        for _i in range(n):
            s += values[rnd.randrange(0, n)]
        samples.append(s / n)

    samples.sort()
    lo_idx = int((alpha / 2) * iters)
    hi_idx = int((1 - alpha / 2) * iters) - 1
    lo_idx = max(0, min(lo_idx, iters - 1))
    hi_idx = max(0, min(hi_idx, iters - 1))
    return samples[lo_idx], samples[hi_idx]


def summarize(values: list[float], *, seed: int = 1337) -> UpliftSummary:
    if not values:
        return UpliftSummary(mean_delta=0.0, ci_low=0.0, ci_high=0.0)
    mean = sum(values) / len(values)
    lo, hi = bootstrap_ci(values, seed=seed)
    return UpliftSummary(mean_delta=mean, ci_low=lo, ci_high=hi)
