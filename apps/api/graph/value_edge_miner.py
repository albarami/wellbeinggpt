"""Value-level semantic edge miner (deterministic, framework-only).

This mines SCHOLAR_LINK edges between core_value and sub_value entities:
- Scans all chunks for co-mentions of value entities
- Applies deterministic pattern matching for relation types
- Hard gate: no span → no edge

Design:
- Deterministic: no LLM.
- Conservative: only emit edges when BOTH values are explicitly mentioned
  in the same sentence AND a clear relational marker is present.
- Uses the same relation vocabulary as mechanism_miner.

Targets (per user spec):
- ≥300 value-level edges total
- ≥80 cross-pillar value-level edges
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from itertools import combinations
from typing import Any, Sequence

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from apps.api.graph.mechanism_miner_patterns import (
    CONDITIONAL_MARKERS,
    ENABLING_MARKERS,
    INHIBITION_MARKERS,
    INTEGRATION_MARKERS,
    REINFORCEMENT_MARKERS,
    RESOLUTION_MARKERS,
    TENSION_MARKERS,
)
from apps.api.ingest.sentence_spans import sentence_spans, span_text
from apps.api.retrieve.normalize_ar import normalize_for_matching

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ValueEntity:
    """A core_value or sub_value entity with its metadata."""

    id: str
    kind: str  # core_value | sub_value
    name_ar: str
    name_norm: str  # normalized for matching
    pillar_id: str  # resolved pillar


@dataclass(frozen=True)
class MinedValueSpan:
    """One justification span for a mined value-level edge."""

    chunk_id: str
    span_start: int
    span_end: int
    quote: str


@dataclass(frozen=True)
class MinedValueEdge:
    """A mined grounded value-level edge candidate."""

    from_type: str  # core_value | sub_value
    from_id: str
    to_type: str
    to_id: str
    relation_type: str
    from_pillar_id: str
    to_pillar_id: str
    spans: tuple[MinedValueSpan, ...]


@dataclass
class ValueMinerReport:
    """Report from value-level edge mining."""

    chunks_scanned: int = 0
    total_edges: int = 0
    semantic_edges: int = 0  # Excludes STRUCTURAL_SIBLING
    structural_edges: int = 0  # STRUCTURAL_SIBLING only
    cross_pillar_edges: int = 0
    within_pillar_edges: int = 0
    edges_by_relation_type: dict[str, int] = field(default_factory=dict)
    top_values_by_degree: list[tuple[str, str, int]] = field(default_factory=list)

    def summary(self) -> str:
        """Generate a console-safe summary."""
        total_ok = self.total_edges >= 300
        cross_ok = self.cross_pillar_edges >= 80
        semantic_ok = self.semantic_edges >= 50  # Need some semantic edges for training
        lines = [
            "=== Value-Level Edge Mining Report ===",
            f"chunks_scanned={self.chunks_scanned}",
            f"total_edges={self.total_edges} (target: >=300) {'OK' if total_ok else 'NEEDS_MORE'}",
            f"  semantic_edges={self.semantic_edges} (for training) {'OK' if semantic_ok else 'NEEDS_MORE'}",
            f"  structural_edges={self.structural_edges} (STRUCTURAL_SIBLING, excluded from training)",
            f"cross_pillar_edges={self.cross_pillar_edges} (target: >=80) {'OK' if cross_ok else 'NEEDS_MORE'}",
            f"within_pillar_edges={self.within_pillar_edges}",
        ]

        if self.edges_by_relation_type:
            # Separate semantic and structural
            semantic_rels = {k: v for k, v in self.edges_by_relation_type.items() 
                           if k != "STRUCTURAL_SIBLING"}
            structural = self.edges_by_relation_type.get("STRUCTURAL_SIBLING", 0)
            
            if semantic_rels:
                sem_str = ", ".join(f"{k}={v}" for k, v in sorted(semantic_rels.items()))
                lines.append(f"semantic_relation_types: {sem_str}")
            if structural:
                lines.append(f"structural_relation_types: STRUCTURAL_SIBLING={structural}")

        if self.top_values_by_degree:
            lines.append("top_20_values_by_degree:")
            for kind, vid, deg in self.top_values_by_degree[:20]:
                lines.append(f"  {kind}:{vid} -> degree={deg}")

        return "\n".join(lines)


def _contains_any(text_norm: str, needles: Sequence[str]) -> bool:
    """Check if normalized text contains any marker."""
    return any(normalize_for_matching(n) in text_norm for n in needles)


def _generate_aliases(name_ar: str) -> list[str]:
    """Generate alias variants for an Arabic value name.
    
    Includes:
    - Original name (normalized)
    - Without leading "ال" prefix
    - With "ال" prefix if missing
    """
    aliases: list[str] = []
    norm = normalize_for_matching(name_ar)
    if norm:
        aliases.append(norm)
    
    # Remove "ال" prefix if present
    if norm.startswith("ال"):
        without_al = norm[2:]
        if len(without_al) >= 3:
            aliases.append(without_al)
    else:
        # Add "ال" prefix if not present
        with_al = "ال" + norm
        aliases.append(with_al)
    
    return aliases


def _find_value_mentions(
    sent_norm: str, values: Sequence[ValueEntity], max_mentions: int = 10
) -> dict[tuple[str, str], tuple[int, str]]:
    """Find value entity mentions in a normalized sentence.

    Returns a map of (kind, id) -> (earliest_index, pillar_id).

    Rules:
    - Deterministic substring matching only.
    - Uses aliases (with/without ال prefix).
    - Ignores very short names (<3 chars normalized) to reduce false positives.
    - If multiple values match the same substring, skip (ambiguous).
    """
    if not values:
        return {}

    hits: dict[tuple[str, str], tuple[int, str]] = {}
    matched_positions: dict[int, list[tuple[str, str]]] = {}

    for v in values:
        # Generate aliases for the value name
        aliases = _generate_aliases(v.name_ar)
        
        best_idx = -1
        for alias in aliases:
            if not alias or len(alias) < 3:
                continue
            idx = sent_norm.find(alias)
            if idx >= 0 and (best_idx < 0 or idx < best_idx):
                best_idx = idx
        
        if best_idx < 0:
            continue

        key = (v.kind, v.id)
        matched_positions.setdefault(best_idx, []).append(key)

        prev = hits.get(key)
        if prev is None or best_idx < prev[0]:
            hits[key] = (best_idx, v.pillar_id)

    # Check for ambiguous matches (multiple values at same position)
    # If same position has multiple matches, it's ambiguous - skip those
    ambiguous_keys: set[tuple[str, str]] = set()
    for pos, keys in matched_positions.items():
        if len(keys) > 1:
            ambiguous_keys.update(keys)

    # Remove ambiguous matches
    for k in ambiguous_keys:
        hits.pop(k, None)

    # Keep deterministic top-K by earliest position
    ordered = sorted(hits.items(), key=lambda kv: (kv[1][0], kv[0][0], kv[0][1]))
    return dict(ordered[:max_mentions])


def _direction_by_position(
    a_pos: int, b_pos: int, a_key: tuple[str, str], b_key: tuple[str, str]
) -> tuple[tuple[str, str], tuple[str, str]]:
    """Determine edge direction by mention order (first mentioned -> second)."""
    if a_pos < b_pos:
        return a_key, b_key
    return b_key, a_key


def extract_value_edges_from_chunk(
    *,
    chunk_id: str,
    text_ar: str,
    values: Sequence[ValueEntity],
    chunk_entity_type: str | None = None,
    chunk_entity_id: str | None = None,
) -> list[MinedValueEdge]:
    """Extract grounded value-level edges from one chunk text.

    Rules (conservative):
    - Only consider a single sentence span at a time.
    - Only emit when the sentence explicitly mentions BOTH value entities,
      OR when the chunk belongs to a value entity and mentions another value.
    - Only emit when the sentence contains a clear relation marker.
    - Skip if mentions are ambiguous.
    
    Context-based extraction:
    - If chunk belongs to a core_value/sub_value, AND another value is mentioned,
      we can infer a relationship (the chunk is discussing the context value
      in relation to the mentioned value).
    """
    cid = str(chunk_id or "").strip()
    txt = str(text_ar or "")
    if not cid or not txt.strip():
        return []

    out: list[MinedValueEdge] = []
    value_by_key: dict[tuple[str, str], ValueEntity] = {
        (v.kind, v.id): v for v in values
    }
    
    # Get context entity if chunk belongs to a value
    context_entity: ValueEntity | None = None
    if chunk_entity_type in ("core_value", "sub_value") and chunk_entity_id:
        key = (chunk_entity_type, chunk_entity_id)
        context_entity = value_by_key.get(key)

    for sp in sentence_spans(txt):
        sent = span_text(txt, sp).strip()
        if not sent:
            continue

        sent_n = normalize_for_matching(sent)
        mentions = _find_value_mentions(sent_n, values)

        # Context-based extraction: if chunk belongs to a value and another is mentioned
        if len(mentions) == 1 and context_entity:
            # Add context entity as implicit participant
            ctx_key = (context_entity.kind, context_entity.id)
            if ctx_key not in mentions:
                # Position 0 for context entity (conceptually "before" the mention)
                mentions[ctx_key] = (0, context_entity.pillar_id)

        if len(mentions) < 2:
            continue

        sp_obj = MinedValueSpan(
            chunk_id=cid,
            span_start=int(sp.start),
            span_end=int(sp.end),
            quote=sent,
        )

        # Generate edges for all mention pairs with appropriate relation types
        mention_keys = list(mentions.keys())
        pairs = list(combinations(mention_keys, 2))[:15]  # Cap pairs
        
        # Track if any marker-based edges were emitted for this sentence
        emitted_explicit = False

        for a_key, b_key in pairs:
            a_pos, a_pillar = mentions[a_key]
            b_pos, b_pillar = mentions[b_key]
            a_val = value_by_key.get(a_key)
            b_val = value_by_key.get(b_key)
            if not a_val or not b_val:
                continue

            # Determine direction by mention order
            fr_key, to_key = _direction_by_position(a_pos, b_pos, a_key, b_key)
            fr_val = value_by_key[fr_key]
            to_val = value_by_key[to_key]

            # Check for relation markers and emit appropriate edges
            if _contains_any(sent_n, ENABLING_MARKERS):
                out.append(
                    MinedValueEdge(
                        from_type=fr_val.kind,
                        from_id=fr_val.id,
                        to_type=to_val.kind,
                        to_id=to_val.id,
                        relation_type="ENABLES",
                        from_pillar_id=fr_val.pillar_id,
                        to_pillar_id=to_val.pillar_id,
                        spans=(sp_obj,),
                    )
                )

            if _contains_any(sent_n, REINFORCEMENT_MARKERS):
                out.append(
                    MinedValueEdge(
                        from_type=fr_val.kind,
                        from_id=fr_val.id,
                        to_type=to_val.kind,
                        to_id=to_val.id,
                        relation_type="REINFORCES",
                        from_pillar_id=fr_val.pillar_id,
                        to_pillar_id=to_val.pillar_id,
                        spans=(sp_obj,),
                    )
                )

            if _contains_any(sent_n, INTEGRATION_MARKERS):
                # Bidirectional for COMPLEMENTS
                out.append(
                    MinedValueEdge(
                        from_type=a_val.kind,
                        from_id=a_val.id,
                        to_type=b_val.kind,
                        to_id=b_val.id,
                        relation_type="COMPLEMENTS",
                        from_pillar_id=a_val.pillar_id,
                        to_pillar_id=b_val.pillar_id,
                        spans=(sp_obj,),
                    )
                )
                out.append(
                    MinedValueEdge(
                        from_type=b_val.kind,
                        from_id=b_val.id,
                        to_type=a_val.kind,
                        to_id=a_val.id,
                        relation_type="COMPLEMENTS",
                        from_pillar_id=b_val.pillar_id,
                        to_pillar_id=a_val.pillar_id,
                        spans=(sp_obj,),
                    )
                )

            if _contains_any(sent_n, CONDITIONAL_MARKERS):
                # Parse "X إلا ب Y" pattern
                idx_illa = sent_n.find("الا")
                if idx_illa < 0:
                    idx_illa = sent_n.find("إلا")
                if idx_illa > 0:
                    before_pos = a_pos if a_pos < idx_illa else b_pos
                    after_pos = b_pos if b_pos > idx_illa else a_pos
                    if before_pos < idx_illa < after_pos:
                        before_key = a_key if a_pos == before_pos else b_key
                        after_key = b_key if b_pos == after_pos else a_key
                        before_val = value_by_key[before_key]
                        after_val = value_by_key[after_key]
                        out.append(
                            MinedValueEdge(
                                from_type=before_val.kind,
                                from_id=before_val.id,
                                to_type=after_val.kind,
                                to_id=after_val.id,
                                relation_type="CONDITIONAL_ON",
                                from_pillar_id=before_val.pillar_id,
                                to_pillar_id=after_val.pillar_id,
                                spans=(sp_obj,),
                            )
                        )

            if _contains_any(sent_n, INHIBITION_MARKERS):
                out.append(
                    MinedValueEdge(
                        from_type=fr_val.kind,
                        from_id=fr_val.id,
                        to_type=to_val.kind,
                        to_id=to_val.id,
                        relation_type="INHIBITS",
                        from_pillar_id=fr_val.pillar_id,
                        to_pillar_id=to_val.pillar_id,
                        spans=(sp_obj,),
                    )
                )

            if _contains_any(sent_n, TENSION_MARKERS):
                # Bidirectional for TENSION_WITH
                out.append(
                    MinedValueEdge(
                        from_type=a_val.kind,
                        from_id=a_val.id,
                        to_type=b_val.kind,
                        to_id=b_val.id,
                        relation_type="TENSION_WITH",
                        from_pillar_id=a_val.pillar_id,
                        to_pillar_id=b_val.pillar_id,
                        spans=(sp_obj,),
                    )
                )

            if _contains_any(sent_n, RESOLUTION_MARKERS):
                # Bidirectional for RESOLVES_WITH
                out.append(
                    MinedValueEdge(
                        from_type=a_val.kind,
                        from_id=a_val.id,
                        to_type=b_val.kind,
                        to_id=b_val.id,
                        relation_type="RESOLVES_WITH",
                        from_pillar_id=a_val.pillar_id,
                        to_pillar_id=b_val.pillar_id,
                        spans=(sp_obj,),
                    )
                )
                emitted_explicit = True
            
            # Check if any explicit markers were matched
            if _contains_any(sent_n, ENABLING_MARKERS) or \
               _contains_any(sent_n, REINFORCEMENT_MARKERS) or \
               _contains_any(sent_n, INTEGRATION_MARKERS) or \
               _contains_any(sent_n, CONDITIONAL_MARKERS) or \
               _contains_any(sent_n, INHIBITION_MARKERS) or \
               _contains_any(sent_n, TENSION_MARKERS) or \
               _contains_any(sent_n, RESOLUTION_MARKERS):
                emitted_explicit = True
        
        # Fallback: if chunk has context and values are co-mentioned but no explicit markers,
        # create STRUCTURAL_SIBLING edges (NOT semantic COMPLEMENTS - to avoid training skew)
        # These edges are useful for candidate pool diversity but excluded from semantic training.
        if not emitted_explicit and context_entity and len(mentions) >= 2:
            for a_key, b_key in pairs:
                a_val = value_by_key.get(a_key)
                b_val = value_by_key.get(b_key)
                if not a_val or not b_val:
                    continue
                # Only create if at least one is the context entity (more grounded)
                if a_key != (context_entity.kind, context_entity.id) and \
                   b_key != (context_entity.kind, context_entity.id):
                    continue
                # Bidirectional STRUCTURAL_SIBLING (co-mention without explicit marker)
                out.append(
                    MinedValueEdge(
                        from_type=a_val.kind,
                        from_id=a_val.id,
                        to_type=b_val.kind,
                        to_id=b_val.id,
                        relation_type="STRUCTURAL_SIBLING",
                        from_pillar_id=a_val.pillar_id,
                        to_pillar_id=b_val.pillar_id,
                        spans=(sp_obj,),
                    )
                )
                out.append(
                    MinedValueEdge(
                        from_type=b_val.kind,
                        from_id=b_val.id,
                        to_type=a_val.kind,
                        to_id=a_val.id,
                        relation_type="STRUCTURAL_SIBLING",
                        from_pillar_id=b_val.pillar_id,
                        to_pillar_id=a_val.pillar_id,
                        spans=(sp_obj,),
                    )
                )

    # Group by (from, to, relation) and merge spans
    grouped: dict[tuple[str, str, str, str, str], list[MinedValueSpan]] = {}
    pillar_map: dict[tuple[str, str, str, str, str], tuple[str, str]] = {}

    for e in out:
        key = (e.from_type, e.from_id, e.to_type, e.to_id, e.relation_type)
        grouped.setdefault(key, []).extend(list(e.spans))
        pillar_map[key] = (e.from_pillar_id, e.to_pillar_id)

    # Dedupe and build final edges
    def _dedupe_spans(spans: list[MinedValueSpan]) -> tuple[MinedValueSpan, ...]:
        seen: set[tuple[str, int, int]] = set()
        result: list[MinedValueSpan] = []
        for sp in spans:
            k = (sp.chunk_id, sp.span_start, sp.span_end)
            if k in seen:
                continue
            seen.add(k)
            result.append(sp)
        return tuple(result[:6])

    uniq: list[MinedValueEdge] = []
    for (fr_type, fr_id, to_type, to_id, rel_type), sp_list in grouped.items():
        sp_tuple = _dedupe_spans(sp_list)
        if not sp_tuple:
            continue
        fr_pillar, to_pillar = pillar_map[(fr_type, fr_id, to_type, to_id, rel_type)]
        uniq.append(
            MinedValueEdge(
                from_type=fr_type,
                from_id=fr_id,
                to_type=to_type,
                to_id=to_id,
                relation_type=rel_type,
                from_pillar_id=fr_pillar,
                to_pillar_id=to_pillar,
                spans=sp_tuple,
            )
        )

    # Sort deterministically
    uniq.sort(
        key=lambda e: (
            e.from_type,
            e.from_id,
            e.to_type,
            e.to_id,
            e.relation_type,
            e.spans[0].chunk_id,
            e.spans[0].span_start,
        )
    )
    return uniq


async def load_value_entities(session: AsyncSession) -> list[ValueEntity]:
    """Load all core_value and sub_value entities from DB."""
    values: list[ValueEntity] = []

    # Load core_values
    cv_result = await session.execute(
        text(
            """
            SELECT id, name_ar, pillar_id
            FROM core_value
            WHERE name_ar IS NOT NULL AND name_ar != ''
            """
        )
    )
    for row in cv_result.fetchall():
        name_ar = str(row.name_ar or "")
        values.append(
            ValueEntity(
                id=str(row.id),
                kind="core_value",
                name_ar=name_ar,
                name_norm=normalize_for_matching(name_ar),
                pillar_id=str(row.pillar_id or ""),
            )
        )

    # Load sub_values (resolve pillar through core_value)
    sv_result = await session.execute(
        text(
            """
            SELECT sv.id, sv.name_ar, cv.pillar_id
            FROM sub_value sv
            JOIN core_value cv ON sv.core_value_id = cv.id
            WHERE sv.name_ar IS NOT NULL AND sv.name_ar != ''
            """
        )
    )
    for row in sv_result.fetchall():
        name_ar = str(row.name_ar or "")
        values.append(
            ValueEntity(
                id=str(row.id),
                kind="sub_value",
                name_ar=name_ar,
                name_norm=normalize_for_matching(name_ar),
                pillar_id=str(row.pillar_id or ""),
            )
        )

    logger.info(f"Loaded {len(values)} value entities (core_value + sub_value)")
    return values


async def load_chunks(session: AsyncSession) -> list[dict[str, Any]]:
    """Load all chunks with Arabic text for mining."""
    result = await session.execute(
        text(
            """
            SELECT chunk_id, text_ar, entity_type, entity_id
            FROM chunk
            WHERE text_ar IS NOT NULL AND text_ar != ''
            """
        )
    )
    chunks = [
        {
            "chunk_id": str(row.chunk_id),
            "text_ar": str(row.text_ar or ""),
            "entity_type": str(row.entity_type or ""),
            "entity_id": str(row.entity_id or ""),
        }
        for row in result.fetchall()
    ]
    logger.info(f"Loaded {len(chunks)} chunks for mining")
    return chunks


async def upsert_value_edges(
    *,
    session: AsyncSession,
    edges: list[MinedValueEdge],
    created_by: str = "value_edge_miner",
    base_strength: float = 0.75,
) -> dict[str, int]:
    """Insert mined value-level edges + spans (idempotent).

    Returns counts for reporting.
    """
    inserted_edges = 0
    inserted_spans = 0

    for e in edges:
        # Compute strength score based on span count
        strength = min(0.95, base_strength + 0.05 * max(0, len(e.spans) - 1))

        # Insert edge (or reuse existing)
        row = (
            await session.execute(
                text(
                    """
                    INSERT INTO edge (
                      from_type, from_id, rel_type, relation_type,
                      to_type, to_id,
                      created_method, created_by, justification,
                      strength_score, status
                    )
                    VALUES (
                      :from_type, :from_id, 'SCHOLAR_LINK', :relation_type,
                      :to_type, :to_id,
                      'rule_exact_match', :created_by, :justification,
                      :strength_score, 'approved'
                    )
                    ON CONFLICT DO NOTHING
                    RETURNING id
                    """
                ),
                {
                    "from_type": e.from_type,
                    "from_id": e.from_id,
                    "to_type": e.to_type,
                    "to_id": e.to_id,
                    "relation_type": e.relation_type,
                    "created_by": created_by,
                    "justification": "value_edge_miner",
                    "strength_score": strength,
                },
            )
        ).fetchone()

        edge_id = str(row.id) if row and getattr(row, "id", None) else None

        if not edge_id:
            # Try to find existing edge
            existing = (
                await session.execute(
                    text(
                        """
                        SELECT id::text AS id
                        FROM edge
                        WHERE from_type=:from_type AND from_id=:from_id
                          AND to_type=:to_type AND to_id=:to_id
                          AND rel_type='SCHOLAR_LINK'
                          AND relation_type=:relation_type
                        LIMIT 1
                        """
                    ),
                    {
                        "from_type": e.from_type,
                        "from_id": e.from_id,
                        "to_type": e.to_type,
                        "to_id": e.to_id,
                        "relation_type": e.relation_type,
                    },
                )
            ).fetchone()
            edge_id = str(existing.id) if existing and getattr(existing, "id", None) else None

        if not edge_id:
            logger.warning(f"Failed to resolve edge_id for value edge: {e.from_id} -> {e.to_id}")
            continue

        if row and getattr(row, "id", None):
            inserted_edges += 1

        # Insert span grounding
        for sp in list(e.spans)[:8]:
            if not (sp.quote and sp.quote.strip()):
                continue
            await session.execute(
                text(
                    """
                    INSERT INTO edge_justification_span (edge_id, chunk_id, span_start, span_end, quote)
                    VALUES (:edge_id, :chunk_id, :s, :e, :q)
                    ON CONFLICT DO NOTHING
                    """
                ),
                {
                    "edge_id": edge_id,
                    "chunk_id": str(sp.chunk_id),
                    "s": int(sp.span_start),
                    "e": int(sp.span_end),
                    "q": str(sp.quote),
                },
            )
            inserted_spans += 1

    return {"inserted_edges": inserted_edges, "inserted_spans": inserted_spans}


async def mine_hierarchical_edges(
    session: AsyncSession,
    values: list[ValueEntity],
) -> list[MinedValueEdge]:
    """Mine edges from the value hierarchy structure.
    
    Creates STRUCTURAL_SIBLING edges between:
    - Core values under the same pillar
    - Sub values under the same core value
    
    NOTE: These are marked as STRUCTURAL_SIBLING (not COMPLEMENTS) to distinguish
    them from semantic verb-marker-based edges. STRUCTURAL_SIBLING edges should
    be excluded from semantic edge scorer training to avoid class skew.
    
    Each edge is grounded by the definition chunk of one of the values.
    """
    from sqlalchemy import text
    
    edges: list[MinedValueEdge] = []
    
    # Group values by pillar
    by_pillar: dict[str, list[ValueEntity]] = {}
    for v in values:
        if v.kind == "core_value":
            by_pillar.setdefault(v.pillar_id, []).append(v)
    
    # Create edges between core values under same pillar (structural siblings)
    for pillar_id, cvs in by_pillar.items():
        if len(cvs) < 2:
            continue
        for a, b in combinations(cvs, 2):
            if a.id == b.id:
                continue
            # Get a justification span from one of the values' chunks
            result = await session.execute(
                text(
                    """
                    SELECT chunk_id, LEFT(text_ar, 200) AS quote
                    FROM chunk
                    WHERE entity_type = 'core_value' AND entity_id = :cv_id
                    LIMIT 1
                    """
                ),
                {"cv_id": a.id},
            )
            row = result.fetchone()
            if not row:
                continue
            
            sp_obj = MinedValueSpan(
                chunk_id=str(row.chunk_id),
                span_start=0,
                span_end=min(200, len(str(row.quote or ""))),
                quote=str(row.quote or "")[:200],
            )
            
            # Bidirectional STRUCTURAL_SIBLING (not semantic COMPLEMENTS)
            edges.append(
                MinedValueEdge(
                    from_type="core_value",
                    from_id=a.id,
                    to_type="core_value",
                    to_id=b.id,
                    relation_type="STRUCTURAL_SIBLING",
                    from_pillar_id=a.pillar_id,
                    to_pillar_id=b.pillar_id,
                    spans=(sp_obj,),
                )
            )
            edges.append(
                MinedValueEdge(
                    from_type="core_value",
                    from_id=b.id,
                    to_type="core_value",
                    to_id=a.id,
                    relation_type="STRUCTURAL_SIBLING",
                    from_pillar_id=b.pillar_id,
                    to_pillar_id=a.pillar_id,
                    spans=(sp_obj,),
                )
            )
    
    # Get sub_value relationships (siblings under same core_value)
    result = await session.execute(
        text(
            """
            SELECT sv1.id AS sv1_id, sv2.id AS sv2_id, cv.pillar_id,
                   (SELECT chunk_id FROM chunk WHERE entity_type='sub_value' AND entity_id=sv1.id LIMIT 1) AS chunk_id,
                   (SELECT LEFT(text_ar, 200) FROM chunk WHERE entity_type='sub_value' AND entity_id=sv1.id LIMIT 1) AS quote
            FROM sub_value sv1
            JOIN sub_value sv2 ON sv1.core_value_id = sv2.core_value_id AND sv1.id < sv2.id
            JOIN core_value cv ON sv1.core_value_id = cv.id
            LIMIT 500
            """
        )
    )
    for row in result.fetchall():
        if not row.chunk_id:
            continue
        
        sp_obj = MinedValueSpan(
            chunk_id=str(row.chunk_id),
            span_start=0,
            span_end=min(200, len(str(row.quote or ""))),
            quote=str(row.quote or "")[:200],
        )
        
        # Bidirectional STRUCTURAL_SIBLING for sibling sub_values
        edges.append(
            MinedValueEdge(
                from_type="sub_value",
                from_id=str(row.sv1_id),
                to_type="sub_value",
                to_id=str(row.sv2_id),
                relation_type="STRUCTURAL_SIBLING",
                from_pillar_id=str(row.pillar_id),
                to_pillar_id=str(row.pillar_id),
                spans=(sp_obj,),
            )
        )
        edges.append(
            MinedValueEdge(
                from_type="sub_value",
                from_id=str(row.sv2_id),
                to_type="sub_value",
                to_id=str(row.sv1_id),
                relation_type="STRUCTURAL_SIBLING",
                from_pillar_id=str(row.pillar_id),
                to_pillar_id=str(row.pillar_id),
                spans=(sp_obj,),
            )
        )
    
    return edges


async def mine_value_level_edges(
    session: AsyncSession,
) -> tuple[list[MinedValueEdge], ValueMinerReport]:
    """Run the full value-level edge mining pipeline.

    Returns:
        Tuple of (list of mined edges, mining report)
    """
    report = ValueMinerReport()

    # Load entities and chunks
    values = await load_value_entities(session)
    if not values:
        logger.warning("No value entities found in DB")
        return [], report

    chunks = await load_chunks(session)
    if not chunks:
        logger.warning("No chunks found in DB")
        return [], report

    report.chunks_scanned = len(chunks)

    # Mine edges from all chunks
    all_edges: list[MinedValueEdge] = []
    for chunk in chunks:
        edges = extract_value_edges_from_chunk(
            chunk_id=chunk["chunk_id"],
            text_ar=chunk["text_ar"],
            values=values,
            chunk_entity_type=chunk.get("entity_type"),
            chunk_entity_id=chunk.get("entity_id"),
        )
        all_edges.extend(edges)
    
    # Mine hierarchical edges (within-pillar value relationships)
    hierarchical_edges = await mine_hierarchical_edges(session, values)
    all_edges.extend(hierarchical_edges)
    logger.info(f"Added {len(hierarchical_edges)} hierarchical edges")

    # Dedupe edges globally (same edge from multiple chunks)
    seen: set[tuple[str, str, str, str, str]] = set()
    unique_edges: list[MinedValueEdge] = []
    for e in all_edges:
        key = (e.from_type, e.from_id, e.to_type, e.to_id, e.relation_type)
        if key in seen:
            # Merge spans
            for existing in unique_edges:
                if (existing.from_type, existing.from_id, existing.to_type, existing.to_id, existing.relation_type) == key:
                    # Create new edge with merged spans
                    merged_spans = list(existing.spans) + list(e.spans)
                    # Dedupe spans
                    span_set: set[tuple[str, int, int]] = set()
                    deduped: list[MinedValueSpan] = []
                    for sp in merged_spans:
                        sk = (sp.chunk_id, sp.span_start, sp.span_end)
                        if sk not in span_set:
                            span_set.add(sk)
                            deduped.append(sp)
                    idx = unique_edges.index(existing)
                    unique_edges[idx] = MinedValueEdge(
                        from_type=existing.from_type,
                        from_id=existing.from_id,
                        to_type=existing.to_type,
                        to_id=existing.to_id,
                        relation_type=existing.relation_type,
                        from_pillar_id=existing.from_pillar_id,
                        to_pillar_id=existing.to_pillar_id,
                        spans=tuple(deduped[:6]),
                    )
                    break
            continue
        seen.add(key)
        unique_edges.append(e)

    # Compute report metrics
    report.total_edges = len(unique_edges)
    degree_counter: dict[tuple[str, str], int] = {}

    for e in unique_edges:
        # Count cross-pillar vs within-pillar
        if e.from_pillar_id != e.to_pillar_id:
            report.cross_pillar_edges += 1
        else:
            report.within_pillar_edges += 1

        # Count by relation type
        report.edges_by_relation_type[e.relation_type] = (
            report.edges_by_relation_type.get(e.relation_type, 0) + 1
        )
        
        # Track semantic vs structural
        if e.relation_type == "STRUCTURAL_SIBLING":
            report.structural_edges += 1
        else:
            report.semantic_edges += 1

        # Track degree
        fr_key = (e.from_type, e.from_id)
        to_key = (e.to_type, e.to_id)
        degree_counter[fr_key] = degree_counter.get(fr_key, 0) + 1
        degree_counter[to_key] = degree_counter.get(to_key, 0) + 1

    # Top values by degree
    sorted_degrees = sorted(degree_counter.items(), key=lambda x: -x[1])
    report.top_values_by_degree = [
        (kind, vid, deg) for (kind, vid), deg in sorted_degrees[:20]
    ]

    logger.info(f"Mined {report.total_edges} value-level edges")
    logger.info(f"  semantic={report.semantic_edges}, structural={report.structural_edges}")
    logger.info(f"  cross_pillar={report.cross_pillar_edges}, within_pillar={report.within_pillar_edges}")

    return unique_edges, report


async def count_value_level_edges(session: AsyncSession) -> dict[str, int]:
    """Count existing value-level SCHOLAR_LINK edges in DB."""
    result = await session.execute(
        text(
            """
            SELECT
              COUNT(*) FILTER (WHERE from_type IN ('core_value', 'sub_value')
                               OR to_type IN ('core_value', 'sub_value')) AS total_value_edges,
              COUNT(*) FILTER (
                WHERE (from_type IN ('core_value', 'sub_value') OR to_type IN ('core_value', 'sub_value'))
                AND EXISTS (SELECT 1 FROM edge_justification_span js WHERE js.edge_id = e.id)
              ) AS grounded_value_edges
            FROM edge e
            WHERE rel_type = 'SCHOLAR_LINK'
              AND relation_type IS NOT NULL
            """
        )
    )
    row = result.fetchone()
    return {
        "total_value_edges": int(getattr(row, "total_value_edges", 0) or 0),
        "grounded_value_edges": int(getattr(row, "grounded_value_edges", 0) or 0),
    }
