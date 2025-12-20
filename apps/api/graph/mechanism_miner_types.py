"""Types for the World Model mechanism edge miner.

Reason:
- Keep the public API small and stable for scripts/tests.
- Enforce the repo rule: keep modules <= 500 lines.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class MinedMechanismSpan:
    """One justification span for a mined mechanism edge."""

    chunk_id: str
    span_start: int
    span_end: int
    quote: str


@dataclass(frozen=True)
class MinedMechanismEdge:
    """A mined grounded mechanism edge candidate.

    Unlike SCHOLAR_LINK edges, mechanism edges can be cross-pillar or within-pillar,
    as long as both endpoints are anchored to existing entities and the relation
    marker is explicitly present in the text.
    """

    from_ref_kind: str  # pillar|core_value|sub_value
    from_ref_id: str
    to_ref_kind: str
    to_ref_id: str
    relation_type: str  # ENABLES|REINFORCES|COMPLEMENTS|CONDITIONAL_ON|INHIBITS|TENSION_WITH|RESOLVES_WITH
    polarity: int  # +1 or -1
    spans: tuple[MinedMechanismSpan, ...]


@dataclass
class MinerTargets:
    """Targets for mechanism edge mining (per adjustment #3)."""

    total_edges_min: int = 150
    cross_pillar_edges_min: int = 40


@dataclass
class MinerReport:
    """Report from mechanism edge mining."""

    chunks_scanned: int = 0
    total_edges: int = 0
    cross_pillar_edges: int = 0
    within_pillar_edges: int = 0
    edges_by_pillar: dict[str, int] = field(default_factory=dict)
    edges_by_relation_type: dict[str, int] = field(default_factory=dict)
    loops_detected: int = 0

    def meets_targets(self, targets: MinerTargets) -> bool:
        return (
            self.total_edges >= targets.total_edges_min
            and self.cross_pillar_edges >= targets.cross_pillar_edges_min
        )

    def summary(self) -> str:
        """Generate a console-safe summary (ASCII-only for Windows shells)."""

        total_ok = self.total_edges >= 150
        cross_ok = self.cross_pillar_edges >= 40
        lines = [
            "=== Mechanism Edge Mining Report ===",
            f"chunks_scanned={self.chunks_scanned}",
            f"total_edges={self.total_edges} (target: >=150) {'OK' if total_ok else 'FAIL'}",
            f"cross_pillar_edges={self.cross_pillar_edges} (target: >=40) {'OK' if cross_ok else 'FAIL'}",
            f"within_pillar_edges={self.within_pillar_edges}",
        ]

        if self.edges_by_pillar:
            pillar_str = ", ".join(
                f"{k}={v}" for k, v in sorted(self.edges_by_pillar.items())
            )
            lines.append(f"edges_by_pillar: {pillar_str}")

        if self.edges_by_relation_type:
            rel_str = ", ".join(
                f"{k}={v}" for k, v in sorted(self.edges_by_relation_type.items())
            )
            lines.append(f"edges_by_relation_type: {rel_str}")

        lines.append(f"loops_detected={self.loops_detected}")
        return "\n".join(lines)

