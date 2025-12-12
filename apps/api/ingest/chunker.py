"""
Chunker Module

Creates citeable chunks from extracted content for:
1. Vector embedding and search
2. Evidence packet generation
3. Citation validation
"""

import hashlib
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional

from apps.api.core.schemas import ChunkType, EntityType
from apps.api.retrieve.normalize_ar import normalize_for_matching


@dataclass
class Chunk:
    """
    A citeable chunk for embedding and retrieval.

    This follows the Evidence Packet schema requirements.
    """

    chunk_id: str
    entity_type: EntityType
    entity_id: str
    chunk_type: ChunkType
    text_ar: str
    text_en: Optional[str]
    source_doc_id: str
    source_anchor: str
    token_count_estimate: int
    refs: list[dict] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.utcnow)


class Chunker:
    """
    Creates chunks from extracted content.

    Chunks are created for:
    - Definitions (المفهوم blocks)
    - Evidence blocks (التأصيل blocks)
    - Commentary (optional)
    """

    def __init__(
        self,
        max_tokens: int = 512,
        overlap_tokens: int = 50,
    ):
        """
        Initialize the chunker.

        Args:
            max_tokens: Maximum tokens per chunk.
            overlap_tokens: Overlap between chunks when splitting.
        """
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens
        # Deterministic IDs (stable across re-ingestion) — no counter needed.

    def chunk_canonical_json(
        self,
        canonical_data: dict[str, Any],
    ) -> list[Chunk]:
        """
        Create chunks from canonical JSON data.

        Args:
            canonical_data: Canonical JSON from extraction.

        Returns:
            List of chunks.
        """
        chunks = []
        source_doc_id = canonical_data.get("meta", {}).get("source_doc_id", "unknown")

        for pillar in canonical_data.get("pillars", []):
            chunks.extend(self._chunk_pillar(pillar, source_doc_id))

        return chunks

    def _chunk_pillar(
        self,
        pillar_data: dict[str, Any],
        source_doc_id: str,
    ) -> list[Chunk]:
        """Create chunks from a pillar."""
        chunks = []
        pillar_id = pillar_data["id"]

        # Chunk pillar description if present
        if pillar_data.get("description_ar"):
            chunk = self._create_chunk(
                entity_type=EntityType.PILLAR,
                entity_id=pillar_id,
                chunk_type=ChunkType.DEFINITION,
                text_ar=pillar_data["description_ar"],
                source_doc_id=source_doc_id,
                source_anchor=self._get_anchor_str(pillar_data.get("source_anchor")),
            )
            chunks.append(chunk)

        # Chunk core values
        for cv_data in pillar_data.get("core_values", []):
            chunks.extend(self._chunk_core_value(cv_data, source_doc_id))

        return chunks

    def _chunk_core_value(
        self,
        cv_data: dict[str, Any],
        source_doc_id: str,
    ) -> list[Chunk]:
        """Create chunks from a core value."""
        chunks = []
        cv_id = cv_data["id"]

        # Chunk definition
        if cv_data.get("definition") and cv_data["definition"].get("text_ar"):
            definition = cv_data["definition"]
            text_ar = definition["text_ar"]

            # Split if too long
            text_chunks = self._split_text_if_needed(text_ar)

            for i, text_chunk in enumerate(text_chunks):
                chunk = self._create_chunk(
                    entity_type=EntityType.CORE_VALUE,
                    entity_id=cv_id,
                    chunk_type=ChunkType.DEFINITION,
                    text_ar=text_chunk,
                    source_doc_id=source_doc_id,
                    source_anchor=self._get_anchor_str(definition.get("source_anchor")),
                    refs=list(definition.get("refs") or []),
                )
                chunks.append(chunk)

        # Chunk evidence
        for evidence in cv_data.get("evidence", []):
            chunk = self._create_evidence_chunk(
                entity_type=EntityType.CORE_VALUE,
                entity_id=cv_id,
                evidence=evidence,
                source_doc_id=source_doc_id,
            )
            if chunk:
                chunks.append(chunk)

        # Chunk sub-values
        for sv_data in cv_data.get("sub_values", []):
            chunks.extend(self._chunk_sub_value(sv_data, source_doc_id))

        return chunks

    def _chunk_sub_value(
        self,
        sv_data: dict[str, Any],
        source_doc_id: str,
    ) -> list[Chunk]:
        """Create chunks from a sub-value."""
        chunks = []
        sv_id = sv_data["id"]

        # Chunk definition
        if sv_data.get("definition") and sv_data["definition"].get("text_ar"):
            definition = sv_data["definition"]
            text_ar = definition["text_ar"]

            text_chunks = self._split_text_if_needed(text_ar)

            for text_chunk in text_chunks:
                chunk = self._create_chunk(
                    entity_type=EntityType.SUB_VALUE,
                    entity_id=sv_id,
                    chunk_type=ChunkType.DEFINITION,
                    text_ar=text_chunk,
                    source_doc_id=source_doc_id,
                    source_anchor=self._get_anchor_str(definition.get("source_anchor")),
                    refs=list(definition.get("refs") or []),
                )
                chunks.append(chunk)

        # Chunk evidence
        for evidence in sv_data.get("evidence", []):
            chunk = self._create_evidence_chunk(
                entity_type=EntityType.SUB_VALUE,
                entity_id=sv_id,
                evidence=evidence,
                source_doc_id=source_doc_id,
            )
            if chunk:
                chunks.append(chunk)

        return chunks

    def _create_evidence_chunk(
        self,
        entity_type: EntityType,
        entity_id: str,
        evidence: dict[str, Any],
        source_doc_id: str,
    ) -> Optional[Chunk]:
        """Create a chunk from an evidence record."""
        text_ar = evidence.get("text_ar", "")
        if not text_ar:
            return None

        # Build refs list
        refs = []
        if evidence.get("evidence_type"):
            refs.append({
                "type": evidence["evidence_type"],
                "ref": evidence.get("ref_raw", ""),
            })

        return self._create_chunk(
            entity_type=entity_type,
            entity_id=entity_id,
            chunk_type=ChunkType.EVIDENCE,
            text_ar=text_ar,
            source_doc_id=source_doc_id,
            source_anchor=self._get_anchor_str(evidence.get("source_anchor")),
            refs=refs,
        )

    def _create_chunk(
        self,
        entity_type: EntityType,
        entity_id: str,
        chunk_type: ChunkType,
        text_ar: str,
        source_doc_id: str,
        source_anchor: str,
        refs: Optional[list[dict]] = None,
    ) -> Chunk:
        """Create a single chunk."""
        # Stable chunk id derived from canonical identity + normalized text.
        norm_text = normalize_for_matching(text_ar)
        basis = f"{entity_type.value}|{entity_id}|{chunk_type.value}|{source_doc_id}|{source_anchor}|{norm_text}"
        chunk_id = "CH_" + hashlib.sha1(basis.encode("utf-8")).hexdigest()[:12]

        return Chunk(
            chunk_id=chunk_id,
            entity_type=entity_type,
            entity_id=entity_id,
            chunk_type=chunk_type,
            text_ar=text_ar,
            text_en=None,  # Nullable per MVP policy
            source_doc_id=source_doc_id,
            source_anchor=source_anchor,
            token_count_estimate=self._estimate_tokens(text_ar),
            refs=refs or [],
        )

    def _split_text_if_needed(self, text: str) -> list[str]:
        """
        Split text into chunks if it exceeds max tokens.

        Uses a simple word-based splitting strategy.
        """
        estimated_tokens = self._estimate_tokens(text)

        if estimated_tokens <= self.max_tokens:
            return [text]

        # Split by sentences or words
        chunks = []
        words = text.split()
        current_chunk = []
        current_tokens = 0

        for word in words:
            word_tokens = self._estimate_tokens(word)

            if current_tokens + word_tokens > self.max_tokens:
                if current_chunk:
                    chunks.append(" ".join(current_chunk))
                # Start new chunk with overlap
                overlap_words = current_chunk[-self.overlap_tokens:] if len(current_chunk) > self.overlap_tokens else []
                current_chunk = overlap_words + [word]
                current_tokens = sum(self._estimate_tokens(w) for w in current_chunk)
            else:
                current_chunk.append(word)
                current_tokens += word_tokens

        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return chunks

    def _estimate_tokens(self, text: str) -> int:
        """
        Estimate token count for text.

        Uses a simple heuristic: ~1.5 tokens per word for Arabic.
        """
        if not text:
            return 0

        words = len(text.split())
        # Arabic tends to have more tokens per word due to morphology
        return int(words * 1.5)

    def _get_anchor_str(self, anchor: Any) -> str:
        """
        Convert stored anchor into the required string form.

        For this phase, we require: "para_<index>".
        """
        if anchor is None:
            return "unknown"
        if isinstance(anchor, str):
            return anchor or "unknown"
        if isinstance(anchor, dict):
            # Back-compat if older anchor dict exists
            if isinstance(anchor.get("source_anchor"), str):
                return anchor["source_anchor"] or "unknown"
            if isinstance(anchor.get("anchor_id"), str):
                return anchor["anchor_id"] or "unknown"
        s = str(anchor)
        return s if s else "unknown"

    def save_chunks_jsonl(self, chunks: list[Chunk], output_path: str) -> None:
        """
        Save chunks to JSONL file.

        Each line is a dict compatible with Evidence Packet schema.
        """
        from pathlib import Path
        import json

        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            for c in chunks:
                f.write(json.dumps(chunk_to_evidence_packet(c), ensure_ascii=False) + "\n")


def chunk_to_dict(chunk: Chunk) -> dict[str, Any]:
    """Convert a chunk to dictionary for storage."""
    return {
        "chunk_id": chunk.chunk_id,
        "entity_type": chunk.entity_type.value,
        "entity_id": chunk.entity_id,
        "chunk_type": chunk.chunk_type.value,
        "text_ar": chunk.text_ar,
        "text_en": chunk.text_en,
        "source_doc_id": chunk.source_doc_id,
        "source_anchor": chunk.source_anchor,
        "token_count_estimate": chunk.token_count_estimate,
        "refs": chunk.refs,
        "created_at": chunk.created_at.isoformat(),
    }


def chunk_to_evidence_packet(chunk: Chunk) -> dict[str, Any]:
    """
    Convert a chunk to the Evidence Packet schema.

    This is the authoritative output format for retrieval.
    """
    return {
        "chunk_id": chunk.chunk_id,
        "entity_type": chunk.entity_type.value,
        "entity_id": chunk.entity_id,
        "chunk_type": chunk.chunk_type.value,
        "text_ar": chunk.text_ar,
        "source_doc_id": chunk.source_doc_id,
        "source_anchor": chunk.source_anchor,
        "refs": chunk.refs,
    }

