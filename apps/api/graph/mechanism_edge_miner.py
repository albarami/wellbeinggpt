"""Mechanism edge miner (public facade).

Reason:
- Keep this file small (<= 500 lines) and re-export the public API.
- Implementation is split into smaller modules:
  - mechanism_miner_patterns.py (constants)
  - mechanism_miner_extract.py (text mining)
  - mechanism_miner_db.py (DB upserts)
  - mechanism_miner_types.py (types/reports)
"""

from __future__ import annotations

from apps.api.graph.mechanism_miner_db import get_or_create_mechanism_node, upsert_mechanism_edges
from apps.api.graph.mechanism_miner_extract import (
    LexiconEntry,
    extract_mechanism_edges_from_chunk,
    is_cross_pillar_edge,
)
from apps.api.graph.mechanism_miner_types import (
    MinedMechanismEdge,
    MinedMechanismSpan,
    MinerReport,
    MinerTargets,
)

__all__ = [
    "LexiconEntry",
    "MinedMechanismEdge",
    "MinedMechanismSpan",
    "MinerReport",
    "MinerTargets",
    "extract_mechanism_edges_from_chunk",
    "get_or_create_mechanism_node",
    "is_cross_pillar_edge",
    "upsert_mechanism_edges",
]
