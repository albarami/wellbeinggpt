"""Mechanism edge extraction (deterministic, evidence-only).

This module mines mechanism edges from chunk text:
- Cross-pillar edges when two+ pillars are explicitly mentioned in the same sentence.
- Within-pillar edges when two+ entities (core/sub values) are explicitly mentioned in the
  same sentence, optionally using the chunk's entity context as an anchor.

Hard gate:
- No edge is emitted unless it has at least one span (sentence quote).
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Iterable, Sequence

from apps.api.core.world_model.schemas import get_default_polarity
from apps.api.ingest.sentence_spans import sentence_spans, span_text
from apps.api.retrieve.normalize_ar import normalize_for_matching

from apps.api.graph.mechanism_miner_patterns import (
    BOUNDARY_MARKERS,
    CONDITIONAL_MARKERS,
    ENABLING_MARKERS,
    INHIBITION_MARKERS,
    INTEGRATION_MARKERS,
    PILLAR_KEYWORDS,
    REINFORCEMENT_MARKERS,
    RESOLUTION_MARKERS,
    TENSION_MARKERS,
)
from apps.api.graph.mechanism_miner_types import MinedMechanismEdge, MinedMechanismSpan


@dataclass(frozen=True)
class LexiconEntry:
    """One lexicon entry used for deterministic mention detection."""

    kind: str  # pillar|core_value|sub_value
    id: str
    name_norm: str


def _contains_any(text_norm: str, needles: Iterable[str]) -> bool:
    """Check if normalized text contains any normalized marker."""
    return any(normalize_for_matching(n) in text_norm for n in needles)


def _pillars_mentioned(text_ar: str) -> list[str]:
    """Extract pillar IDs mentioned in text."""
    t = normalize_for_matching(text_ar or "")
    found: list[str] = []
    for k, pid in PILLAR_KEYWORDS.items():
        if k and (k in t) and (pid not in found):
            found.append(pid)
    return found


def _first_pillar_keyword_index(text_norm: str, pid: str) -> int:
    """Return earliest keyword index for a pillar id (normalized)."""
    best = -1
    for k, p in PILLAR_KEYWORDS.items():
        if p != pid:
            continue
        kk = normalize_for_matching(k)
        if not kk:
            continue
        i = text_norm.find(kk)
        if i >= 0 and (best < 0 or i < best):
            best = i
    return best


def _direction_by_order(text_norm: str, a_pid: str, b_pid: str) -> tuple[str, str] | None:
    """Determine pillar edge direction by mention order in text."""
    ia = _first_pillar_keyword_index(text_norm, a_pid)
    ib = _first_pillar_keyword_index(text_norm, b_pid)
    if ia < 0 or ib < 0 or ia == ib:
        return None
    return (a_pid, b_pid) if ia < ib else (b_pid, a_pid)


def _dedupe_spans(spans: Sequence[MinedMechanismSpan]) -> tuple[MinedMechanismSpan, ...]:
    """Deduplicate spans by (chunk_id, span_start, span_end)."""
    seen: set[tuple[str, int, int]] = set()
    out: list[MinedMechanismSpan] = []
    for sp in spans:
        key = (sp.chunk_id, int(sp.span_start), int(sp.span_end))
        if key in seen:
            continue
        seen.add(key)
        out.append(sp)
    return tuple(out)


def _extract_boundary_spans(text_ar: str, chunk_id: str) -> list[MinedMechanismSpan]:
    """Extract boundary/limits sentences as supplementary spans."""
    out: list[MinedMechanismSpan] = []
    for sp in sentence_spans(text_ar or ""):
        sent = span_text(text_ar, sp).strip()
        if not sent:
            continue
        sn = normalize_for_matching(sent)
        if not _contains_any(sn, BOUNDARY_MARKERS):
            continue
        out.append(
            MinedMechanismSpan(
                chunk_id=chunk_id,
                span_start=int(sp.start),
                span_end=int(sp.end),
                quote=sent,
            )
        )
    return out


def _find_mentions(
    *, sent_norm: str, lexicon: Sequence[LexiconEntry] | None, max_mentions: int = 5
) -> dict[tuple[str, str], int]:
    """Find entity mentions in a normalized sentence.

    Returns a map of (kind, id) -> earliest index.

    Reason:
    - Deterministic substring matching only.
    - Conservative: ignores very short names to reduce false positives.
    """
    if not lexicon:
        return {}

    hits: dict[tuple[str, str], int] = {}
    for e in lexicon:
        nn = e.name_norm
        if not nn or len(nn) < 4:
            continue
        idx = sent_norm.find(nn)
        if idx < 0:
            continue
        key = (e.kind, e.id)
        prev = hits.get(key)
        if prev is None or idx < prev:
            hits[key] = idx

    # Keep deterministic top-K by earliest position, then by key
    ordered = sorted(hits.items(), key=lambda kv: (kv[1], kv[0][0], kv[0][1]))
    return dict(ordered[: max_mentions])


def _emit_pair_edges(
    *,
    out: list[MinedMechanismEdge],
    nodes_with_pos: dict[tuple[str, str], int],
    relation_type: str,
    spans: tuple[MinedMechanismSpan, ...],
    bidirectional: bool,
) -> None:
    """Emit edges between node pairs using mention order for direction."""
    if len(nodes_with_pos) < 2:
        return

    nodes = list(nodes_with_pos.keys())
    for (a_kind, a_id), (b_kind, b_id) in list(combinations(nodes, 2))[:12]:
        if (a_kind, a_id) == (b_kind, b_id):
            continue
        ia = nodes_with_pos.get((a_kind, a_id), -1)
        ib = nodes_with_pos.get((b_kind, b_id), -1)
        if ia == ib:
            continue

        if bidirectional:
            out.append(
                MinedMechanismEdge(
                    from_ref_kind=a_kind,
                    from_ref_id=a_id,
                    to_ref_kind=b_kind,
                    to_ref_id=b_id,
                    relation_type=relation_type,
                    polarity=get_default_polarity(relation_type),
                    spans=spans,
                )
            )
            out.append(
                MinedMechanismEdge(
                    from_ref_kind=b_kind,
                    from_ref_id=b_id,
                    to_ref_kind=a_kind,
                    to_ref_id=a_id,
                    relation_type=relation_type,
                    polarity=get_default_polarity(relation_type),
                    spans=spans,
                )
            )
        else:
            fr, to = ((a_kind, a_id), (b_kind, b_id)) if ia < ib else ((b_kind, b_id), (a_kind, a_id))
            out.append(
                MinedMechanismEdge(
                    from_ref_kind=fr[0],
                    from_ref_id=fr[1],
                    to_ref_kind=to[0],
                    to_ref_id=to[1],
                    relation_type=relation_type,
                    polarity=get_default_polarity(relation_type),
                    spans=spans,
                )
            )


def extract_mechanism_edges_from_chunk(
    *,
    chunk_id: str,
    text_ar: str,
    entity_type: str | None = None,
    entity_id: str | None = None,
    lexicon: Sequence[LexiconEntry] | None = None,
) -> list[MinedMechanismEdge]:
    """Extract grounded mechanism edges from one chunk text.

    Args:
        chunk_id: Chunk identifier.
        text_ar: Arabic text content.
        entity_type: Optional chunk entity type (pillar/core_value/sub_value).
        entity_id: Optional chunk entity id.
        lexicon: Optional mention lexicon (core/sub names) for within-pillar mining.
    """
    cid = str(chunk_id or "").strip()
    txt = str(text_ar or "")
    if not cid or not txt.strip():
        return []

    out: list[MinedMechanismEdge] = []

    for sp in sentence_spans(txt):
        sent = span_text(txt, sp).strip()
        if not sent:
            continue

        sent_n = normalize_for_matching(sent)
        pillars = _pillars_mentioned(sent)

        sp_obj = MinedMechanismSpan(
            chunk_id=cid,
            span_start=int(sp.start),
            span_end=int(sp.end),
            quote=sent,
        )
        spans_tuple = (sp_obj,)

        # ---------------------------------------------------------------------
        # Cross-pillar edges: require explicit pillar mentions in the sentence.
        # ---------------------------------------------------------------------
        if len(pillars) >= 2:
            if _contains_any(sent_n, ENABLING_MARKERS):
                for a, b in list(combinations(pillars[:5], 2))[:10]:
                    d = _direction_by_order(sent_n, a, b)
                    if not d:
                        continue
                    fr, to = d
                    out.append(
                        MinedMechanismEdge(
                            from_ref_kind="pillar",
                            from_ref_id=fr,
                            to_ref_kind="pillar",
                            to_ref_id=to,
                            relation_type="ENABLES",
                            polarity=get_default_polarity("ENABLES"),
                            spans=spans_tuple,
                        )
                    )

            if _contains_any(sent_n, REINFORCEMENT_MARKERS):
                for a, b in list(combinations(pillars[:5], 2))[:10]:
                    d = _direction_by_order(sent_n, a, b)
                    if not d:
                        continue
                    fr, to = d
                    out.append(
                        MinedMechanismEdge(
                            from_ref_kind="pillar",
                            from_ref_id=fr,
                            to_ref_kind="pillar",
                            to_ref_id=to,
                            relation_type="REINFORCES",
                            polarity=get_default_polarity("REINFORCES"),
                            spans=spans_tuple,
                        )
                    )

            if _contains_any(sent_n, INTEGRATION_MARKERS):
                for a, b in list(combinations(pillars[:5], 2))[:10]:
                    if a == b:
                        continue
                    out.append(
                        MinedMechanismEdge(
                            from_ref_kind="pillar",
                            from_ref_id=a,
                            to_ref_kind="pillar",
                            to_ref_id=b,
                            relation_type="COMPLEMENTS",
                            polarity=get_default_polarity("COMPLEMENTS"),
                            spans=spans_tuple,
                        )
                    )
                    out.append(
                        MinedMechanismEdge(
                            from_ref_kind="pillar",
                            from_ref_id=b,
                            to_ref_kind="pillar",
                            to_ref_id=a,
                            relation_type="COMPLEMENTS",
                            polarity=get_default_polarity("COMPLEMENTS"),
                            spans=spans_tuple,
                        )
                    )

            if _contains_any(sent_n, CONDITIONAL_MARKERS):
                idx_illa = sent_n.find("الا")
                if idx_illa < 0:
                    idx_illa = sent_n.find("إلا")
                before = sent_n[:idx_illa] if idx_illa > 0 else sent_n
                after = sent_n[idx_illa:] if idx_illa > 0 else sent_n
                before_p = [p for p in pillars if _first_pillar_keyword_index(before, p) >= 0]
                after_p = [p for p in pillars if _first_pillar_keyword_index(after, p) >= 0]
                if before_p and after_p and before_p[0] != after_p[0]:
                    out.append(
                        MinedMechanismEdge(
                            from_ref_kind="pillar",
                            from_ref_id=before_p[0],
                            to_ref_kind="pillar",
                            to_ref_id=after_p[0],
                            relation_type="CONDITIONAL_ON",
                            polarity=get_default_polarity("CONDITIONAL_ON"),
                            spans=spans_tuple,
                        )
                    )

            if _contains_any(sent_n, INHIBITION_MARKERS):
                for a, b in list(combinations(pillars[:5], 2))[:10]:
                    d = _direction_by_order(sent_n, a, b)
                    if not d:
                        continue
                    fr, to = d
                    out.append(
                        MinedMechanismEdge(
                            from_ref_kind="pillar",
                            from_ref_id=fr,
                            to_ref_kind="pillar",
                            to_ref_id=to,
                            relation_type="INHIBITS",
                            polarity=get_default_polarity("INHIBITS"),
                            spans=spans_tuple,
                        )
                    )

            if _contains_any(sent_n, TENSION_MARKERS):
                for a, b in list(combinations(pillars[:5], 2))[:10]:
                    if a == b:
                        continue
                    out.append(
                        MinedMechanismEdge(
                            from_ref_kind="pillar",
                            from_ref_id=a,
                            to_ref_kind="pillar",
                            to_ref_id=b,
                            relation_type="TENSION_WITH",
                            polarity=get_default_polarity("TENSION_WITH"),
                            spans=spans_tuple,
                        )
                    )

            if _contains_any(sent_n, RESOLUTION_MARKERS):
                for a, b in list(combinations(pillars[:5], 2))[:10]:
                    if a == b:
                        continue
                    out.append(
                        MinedMechanismEdge(
                            from_ref_kind="pillar",
                            from_ref_id=a,
                            to_ref_kind="pillar",
                            to_ref_id=b,
                            relation_type="RESOLVES_WITH",
                            polarity=get_default_polarity("RESOLVES_WITH"),
                            spans=spans_tuple,
                        )
                    )

            continue  # Do not also mine within-pillar from multi-pillar sentences.

        # ---------------------------------------------------------------------
        # Within-pillar/entity edges: mine only when endpoints are mentioned.
        # ---------------------------------------------------------------------
        nodes_with_pos: dict[tuple[str, str], int] = {}

        # Pillar mention (if exactly one is mentioned)
        if len(pillars) == 1:
            pid = pillars[0]
            nodes_with_pos[("pillar", pid)] = _first_pillar_keyword_index(sent_n, pid)

        # Mention lexicon (core/sub values by name)
        nodes_with_pos.update(_find_mentions(sent_norm=sent_n, lexicon=lexicon))

        # Context node (chunk entity) as an anchor for within-pillar extraction.
        if entity_type in ("core_value", "sub_value") and entity_id:
            key = (entity_type, str(entity_id))
            if key not in nodes_with_pos:
                # Place the anchor early so direction-by-order works deterministically.
                nodes_with_pos[key] = 0

        if _contains_any(sent_n, ENABLING_MARKERS):
            _emit_pair_edges(
                out=out,
                nodes_with_pos=nodes_with_pos,
                relation_type="ENABLES",
                spans=spans_tuple,
                bidirectional=False,
            )

        if _contains_any(sent_n, REINFORCEMENT_MARKERS):
            _emit_pair_edges(
                out=out,
                nodes_with_pos=nodes_with_pos,
                relation_type="REINFORCES",
                spans=spans_tuple,
                bidirectional=False,
            )

        if _contains_any(sent_n, INTEGRATION_MARKERS):
            _emit_pair_edges(
                out=out,
                nodes_with_pos=nodes_with_pos,
                relation_type="COMPLEMENTS",
                spans=spans_tuple,
                bidirectional=True,
            )

        if _contains_any(sent_n, INHIBITION_MARKERS):
            _emit_pair_edges(
                out=out,
                nodes_with_pos=nodes_with_pos,
                relation_type="INHIBITS",
                spans=spans_tuple,
                bidirectional=False,
            )

        if _contains_any(sent_n, TENSION_MARKERS):
            _emit_pair_edges(
                out=out,
                nodes_with_pos=nodes_with_pos,
                relation_type="TENSION_WITH",
                spans=spans_tuple,
                bidirectional=True,
            )

        if _contains_any(sent_n, RESOLUTION_MARKERS):
            _emit_pair_edges(
                out=out,
                nodes_with_pos=nodes_with_pos,
                relation_type="RESOLVES_WITH",
                spans=spans_tuple,
                bidirectional=True,
            )

        # CONDITIONAL_ON is intentionally not generalized here yet; it's easy to over-fire.

    # Group by (from, to, relation) and merge spans
    grouped: dict[tuple[str, str, str, str, str], list[MinedMechanismSpan]] = {}
    for e in out:
        key = (e.from_ref_kind, e.from_ref_id, e.to_ref_kind, e.to_ref_id, e.relation_type)
        grouped.setdefault(key, []).extend(list(e.spans))

    # Add boundary spans as supplementary evidence (max 2)
    boundaries = _extract_boundary_spans(txt, cid)
    for key, sp_list in list(grouped.items()):
        sp_list.extend(boundaries[:2])
        grouped[key] = sp_list

    # Build final edges with deterministic ordering
    uniq: list[MinedMechanismEdge] = []
    for (fr_kind, fr_id, to_kind, to_id, rel_type), sp_list in grouped.items():
        sp_tuple = _dedupe_spans(sp_list)[:6]
        if not sp_tuple:
            continue  # Hard gate
        uniq.append(
            MinedMechanismEdge(
                from_ref_kind=fr_kind,
                from_ref_id=fr_id,
                to_ref_kind=to_kind,
                to_ref_id=to_id,
                relation_type=rel_type,
                polarity=get_default_polarity(rel_type),
                spans=sp_tuple,
            )
        )

    uniq.sort(
        key=lambda e: (
            e.from_ref_kind,
            e.from_ref_id,
            e.to_ref_kind,
            e.to_ref_id,
            e.relation_type,
            e.spans[0].chunk_id,
            e.spans[0].span_start,
        )
    )
    return uniq


def is_cross_pillar_edge(edge: MinedMechanismEdge) -> bool:
    """Check if edge connects two different pillars."""
    if edge.from_ref_kind != "pillar" or edge.to_ref_kind != "pillar":
        return False
    return edge.from_ref_id != edge.to_ref_id

