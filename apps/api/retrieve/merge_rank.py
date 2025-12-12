"""
Merge and Rank Module

Merges results from multiple retrieval sources and ranks them
using a deterministic scoring policy.
"""

from dataclasses import dataclass, field
from typing import Any, Optional

from apps.api.core.schemas import ChunkType, EntityType


@dataclass
class ScoredPacket:
    """An evidence packet with a relevance score."""

    packet: dict[str, Any]
    score: float
    sources: list[str] = field(default_factory=list)  # sql, vector, graph


@dataclass
class MergeResult:
    """Result of merging evidence from multiple sources."""

    evidence_packets: list[dict[str, Any]]
    total_found: int
    sources_used: list[str]
    has_definition: bool
    has_evidence: bool


class MergeRanker:
    """
    Merges and ranks evidence packets from multiple sources.

    Implements the deterministic scoring policy from the plan:
    1. Direct match to pillar/value names (highest)
    2. Evidence presence
    3. Vector similarity
    4. Recency/version priority
    """

    def __init__(
        self,
        max_packets: int = 10,
        require_definition: bool = True,
        require_evidence: bool = False,
    ):
        """
        Initialize the merge ranker.

        Args:
            max_packets: Maximum packets to return.
            require_definition: Ensure at least one definition.
            require_evidence: Ensure at least one evidence chunk.
        """
        self.max_packets = max_packets
        self.require_definition = require_definition
        self.require_evidence = require_evidence

    def merge(
        self,
        sql_results: list[dict[str, Any]],
        vector_results: list[dict[str, Any]],
        graph_results: list[dict[str, Any]],
        resolved_entities: Optional[list[dict]] = None,
    ) -> MergeResult:
        """
        Merge results from multiple sources.

        Args:
            sql_results: Results from SQL entity lookup.
            vector_results: Results from vector similarity search.
            graph_results: Results from graph expansion.
            resolved_entities: Resolved entity IDs from query.

        Returns:
            MergeResult with deduplicated and ranked packets.
        """
        # Collect all packets with their sources
        all_packets: dict[str, ScoredPacket] = {}

        # Process SQL results (highest base score)
        for packet in sql_results:
            chunk_id = packet["chunk_id"]
            score = self._base_score(packet, resolved_entities)
            score += 0.3  # Bonus for SQL match

            if chunk_id in all_packets:
                all_packets[chunk_id].score = max(all_packets[chunk_id].score, score)
                all_packets[chunk_id].sources.append("sql")
            else:
                all_packets[chunk_id] = ScoredPacket(
                    packet=packet,
                    score=score,
                    sources=["sql"],
                )

        # Process vector results
        for packet in vector_results:
            chunk_id = packet["chunk_id"]
            similarity = packet.get("similarity", 0.5)
            score = self._base_score(packet, resolved_entities)
            score += similarity * 0.2  # Scaled by similarity

            if chunk_id in all_packets:
                all_packets[chunk_id].score = max(all_packets[chunk_id].score, score)
                all_packets[chunk_id].sources.append("vector")
            else:
                all_packets[chunk_id] = ScoredPacket(
                    packet=packet,
                    score=score,
                    sources=["vector"],
                )

        # Process graph results
        for packet in graph_results:
            chunk_id = packet["chunk_id"]
            depth = packet.get("depth", 1)
            score = self._base_score(packet, resolved_entities)
            score += 0.1 / depth  # Lower score for distant nodes

            if chunk_id in all_packets:
                all_packets[chunk_id].score = max(all_packets[chunk_id].score, score)
                all_packets[chunk_id].sources.append("graph")
            else:
                all_packets[chunk_id] = ScoredPacket(
                    packet=packet,
                    score=score,
                    sources=["graph"],
                )

        # Sort by score
        sorted_packets = sorted(
            all_packets.values(),
            key=lambda p: p.score,
            reverse=True,
        )

        # Ensure diversity requirements
        final_packets = self._ensure_diversity(sorted_packets)

        # Limit to max packets
        final_packets = final_packets[: self.max_packets]

        # Build result
        evidence_packets = [p.packet for p in final_packets]
        sources_used = list(set(
            source
            for p in final_packets
            for source in p.sources
        ))

        has_definition = any(
            p["chunk_type"] == ChunkType.DEFINITION.value
            or p["chunk_type"] == "definition"
            for p in evidence_packets
        )

        has_evidence = any(
            p["chunk_type"] == ChunkType.EVIDENCE.value
            or p["chunk_type"] == "evidence"
            for p in evidence_packets
        )

        return MergeResult(
            evidence_packets=evidence_packets,
            total_found=len(all_packets),
            sources_used=sources_used,
            has_definition=has_definition,
            has_evidence=has_evidence,
        )

    def _base_score(
        self,
        packet: dict[str, Any],
        resolved_entities: Optional[list[dict]] = None,
    ) -> float:
        """
        Calculate base score for a packet.

        Args:
            packet: Evidence packet.
            resolved_entities: Resolved entities from query.

        Returns:
            Base score.
        """
        score = 0.5  # Default score

        # Boost for entity type
        entity_type = packet.get("entity_type", "")
        if entity_type == EntityType.SUB_VALUE.value or entity_type == "sub_value":
            score += 0.1  # More specific = higher score
        elif entity_type == EntityType.CORE_VALUE.value or entity_type == "core_value":
            score += 0.05

        # Boost for chunk type
        chunk_type = packet.get("chunk_type", "")
        if chunk_type == ChunkType.DEFINITION.value or chunk_type == "definition":
            score += 0.15  # Definitions are valuable
        elif chunk_type == ChunkType.EVIDENCE.value or chunk_type == "evidence":
            score += 0.1  # Evidence is also valuable

        # Boost for direct entity match
        if resolved_entities:
            entity_id = packet.get("entity_id", "")
            for resolved in resolved_entities:
                if resolved.get("entity_id") == entity_id:
                    score += 0.2 * resolved.get("confidence", 1.0)
                    break

        # Boost for having references
        refs = packet.get("refs", [])
        if refs:
            score += 0.1

        return min(score, 1.0)  # Cap at 1.0

    def _ensure_diversity(
        self,
        sorted_packets: list[ScoredPacket],
    ) -> list[ScoredPacket]:
        """
        Ensure diversity requirements are met.

        Args:
            sorted_packets: Packets sorted by score.

        Returns:
            Packets with diversity ensured.
        """
        result = []
        has_definition = False
        has_evidence = False

        # First pass: add high-scoring packets
        for p in sorted_packets:
            chunk_type = p.packet.get("chunk_type", "")

            if chunk_type in (ChunkType.DEFINITION.value, "definition"):
                has_definition = True
            elif chunk_type in (ChunkType.EVIDENCE.value, "evidence"):
                has_evidence = True

            result.append(p)

            if len(result) >= self.max_packets:
                break

        # Second pass: ensure requirements if not met
        if self.require_definition and not has_definition:
            # Find a definition packet not yet included
            for p in sorted_packets:
                chunk_type = p.packet.get("chunk_type", "")
                if chunk_type in (ChunkType.DEFINITION.value, "definition"):
                    if p not in result:
                        result.insert(0, p)  # Add at beginning
                        break

        if self.require_evidence and not has_evidence:
            for p in sorted_packets:
                chunk_type = p.packet.get("chunk_type", "")
                if chunk_type in (ChunkType.EVIDENCE.value, "evidence"):
                    if p not in result:
                        result.insert(1, p)  # Add near beginning
                        break

        return result


def create_evidence_bundle(
    packets: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Create a properly formatted evidence bundle.

    Ensures all packets follow the Evidence Packet schema.

    Args:
        packets: Raw packets from retrieval.

    Returns:
        Formatted evidence packets.
    """
    bundle = []

    for packet in packets:
        formatted = {
            "chunk_id": packet.get("chunk_id", ""),
            "entity_type": packet.get("entity_type", ""),
            "entity_id": packet.get("entity_id", ""),
            "chunk_type": packet.get("chunk_type", ""),
            "text_ar": packet.get("text_ar", ""),
            "source_doc_id": packet.get("source_doc_id", ""),
            "source_anchor": packet.get("source_anchor", ""),
            "refs": packet.get("refs", []),
        }
        bundle.append(formatted)

    return bundle

