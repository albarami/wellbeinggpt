"""Framework semantic edge miner (deterministic, framework-only).

This converts explicit cross-pillar statements already present in the framework chunks
into grounded semantic edges:
- edge.rel_type = 'SCHOLAR_LINK'
- edge.relation_type = ENABLES / REINFORCES / ... (controlled vocabulary)
- edge_justification_span rows (chunk_id + offsets + quote) are mandatory

Hard gate:
- If we cannot produce at least one span per edge, we do not insert that edge.

Design:
- Deterministic: no LLM.
- Conservative: only emit edges when the sentence explicitly mentions both pillars
  and contains a clear relational marker.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Iterable, Sequence

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from apps.api.ingest.sentence_spans import sentence_spans, span_text
from apps.api.retrieve.normalize_ar import normalize_for_matching


@dataclass(frozen=True)
class MinedEdgeSpan:
    """One justification span (sentence-like) for a mined edge."""

    chunk_id: str
    span_start: int
    span_end: int
    quote: str


@dataclass(frozen=True)
class MinedEdge:
    """
    A mined grounded edge candidate.

    Note: edges can have N justification spans (multi-span stitching).
    """

    from_pillar_id: str
    to_pillar_id: str
    relation_type: str  # ENABLES|REINFORCES|COMPLEMENTS|CONDITIONAL_ON|TENSION_WITH|RESOLVES_WITH
    spans: tuple[MinedEdgeSpan, ...]


# Align with scholar_reasoning_edge_fallback pillar ids.
_PILLAR_KEYWORDS: dict[str, str] = {
    "روحي": "P001",
    "الروحية": "P001",
    "الحياة الروحية": "P001",
    "الروح": "P001",
    "روح": "P001",
    "ايمان": "P001",
    "الإيمان": "P001",
    "القلب": "P001",
    "قلب": "P001",
    "اخلاقي": "P001",
    "الأخلاقية": "P001",
    "اخلاقية": "P001",
    "عاطفي": "P002",
    "العاطفية": "P002",
    "الحياة العاطفية": "P002",
    "وجداني": "P002",
    "الوجدانية": "P002",
    "وجدانية": "P002",
    "العاطفة": "P002",
    "عاطفة": "P002",
    "مشاعر": "P002",
    "نفس": "P002",
    "نفسي": "P002",
    "النفسية": "P002",
    "نفسية": "P002",
    "انفعالي": "P002",
    "الانفعالية": "P002",
    "فكري": "P003",
    "الفكرية": "P003",
    "الحياة الفكرية": "P003",
    "عقلي": "P003",
    "العقلية": "P003",
    "عقلية": "P003",
    "العقل": "P003",
    "معرفي": "P003",
    "المعرفية": "P003",
    "بدني": "P004",
    "البدنية": "P004",
    "الجسد": "P004",
    "الجسم": "P004",
    "جسم": "P004",
    "بدن": "P004",
    "الحياة البدنية": "P004",
    "الصحة": "P004",
    "صحي": "P004",
    "عافية": "P004",
    "اجتماعي": "P005",
    "الاجتماعية": "P005",
    "الحياة الاجتماعية": "P005",
    "المجتمع": "P005",
    "مجتمعي": "P005",
    "العلاقات الاجتماعية": "P005",
    "اسري": "P005",
    "الاسرية": "P005",
    "أسري": "P005",
    "جماعي": "P005",
    "الجماعية": "P005",
    "علاقات": "P005",
}


def _pillars_mentioned(text_ar: str) -> list[str]:
    t = normalize_for_matching(text_ar or "")
    found: list[str] = []
    for k, pid in _PILLAR_KEYWORDS.items():
        if k and (k in t) and (pid not in found):
            found.append(pid)
    return found


def _contains_any(text_norm: str, needles: Iterable[str]) -> bool:
    return any(n in text_norm for n in needles)


def _first_index(text_norm: str, pid: str) -> int:
    """Return earliest keyword index for a pillar id (normalized)."""

    best = -1
    for k, p in _PILLAR_KEYWORDS.items():
        if p != pid:
            continue
        kk = normalize_for_matching(k)
        if not kk:
            continue
        i = text_norm.find(kk)
        if i >= 0 and (best < 0 or i < best):
            best = i
    return best


def _direct_by_order(text_norm: str, a: str, b: str) -> tuple[str, str] | None:
    ia = _first_index(text_norm, a)
    ib = _first_index(text_norm, b)
    if ia < 0 or ib < 0 or ia == ib:
        return None
    return (a, b) if ia < ib else (b, a)


def _dedupe_spans(spans: Sequence[MinedEdgeSpan]) -> tuple[MinedEdgeSpan, ...]:
    seen: set[tuple[str, int, int]] = set()
    out: list[MinedEdgeSpan] = []
    for sp in spans:
        key = (sp.chunk_id, int(sp.span_start), int(sp.span_end))
        if key in seen:
            continue
        seen.add(key)
        out.append(sp)
    return tuple(out)


def _boundary_sentences(*, text_ar: str) -> list[MinedEdgeSpan]:
    """
    Extract boundary/limits sentences (even if not pillar-specific).

    Reason:
    - The framework often states general "ضوابط/ميزان/تحذير" language without repeating pillar keywords.
    - We attach these as extra spans later when (a) the sentence mentions one of the edge pillars,
      or (b) the sentence is generic (mentions no pillar keywords) and we still want a conservative
      "حدود عامة" span for scholar-honest answers.
    """

    boundary_markers = [
        "ضوابط",
        "حدود",
        "ميزان",
        "انحراف",
        "افراط",
        "إفراط",
        "تفريط",
        "لا ينبغي",
        "لا يجوز",
        "لا يصح",
        "لا يصح أن",
        "تحذير",
        "تنبيه",
        "محاذير",
        # Conditional constraint language often encodes limits.
        "لا يتحقق",
        "لا يكتمل",
        "لا يتم",
        "متوقف على",
        "مشروط",
        "شرط",
        "لا بد",
        "ضرورة",
    ]
    out: list[MinedEdgeSpan] = []
    for sp in sentence_spans(text_ar or ""):
        sent = span_text(text_ar, sp).strip()
        if not sent:
            continue
        sn = normalize_for_matching(sent)
        if not _contains_any(sn, [normalize_for_matching(x) for x in boundary_markers]):
            continue
        out.append(MinedEdgeSpan(chunk_id="", span_start=int(sp.start), span_end=int(sp.end), quote=sent))
    return out


def extract_semantic_edges_from_chunk(
    *,
    chunk_id: str,
    text_ar: str,
) -> list[MinedEdge]:
    """Extract grounded cross-pillar edges from one chunk text.

    Rules (conservative):
    - Only consider a single sentence span at a time.
    - Only emit when the sentence explicitly mentions BOTH pillars (via keywords).
    - Only emit when the sentence contains a clear relation marker.

    Currently supported patterns (framework-only, Arabic-first):
    - Physical -> {Spiritual, Emotional, Intellectual, Social} as ENABLES
      when a sentence describes the body as a means to release/capacitate other domains.
    - Physical <-> Spiritual as REINFORCES when a sentence explicitly says no separation
      between body and soul.
    - Spiritual -> Physical as REINFORCES when a sentence explicitly links heart/faith strength
      to the believer's bodily strength (only if physical is also mentioned).
    """

    cid = str(chunk_id or "").strip()
    txt = str(text_ar or "")
    if not cid or not txt.strip():
        return []

    out: list[MinedEdge] = []
    spans = sentence_spans(txt)
    for sp in spans:
        sent = span_text(txt, sp).strip()
        if not sent:
            continue

        sent_n = normalize_for_matching(sent)
        pillars = _pillars_mentioned(sent)
        if len(pillars) < 2:
            continue

        # Marker sets (normalized).
        enabling_markers = [
            "وسيلة",
            "يمك",
            "ممكن",
            "مُمك",
            "مُعين",
            "يعين",
            "ينعكس",
            "يؤثر",
            "يقود",
            "يثمر",
            "ينتج",
            "يدعم",
            "يساند",
            "يعضد",
            "يعزز",
            "يسهم",
            "يحقق",
            "يؤدي",
            "إطلاق",
            "اطلاق",
            "تحرير",
            "تهيئة",
            "يطلق",
            "تفعيل",
            "يحرر",
        ]
        integration_markers = [
            "تكامل",
            "ترابط",
            "منظومة",
            "متكامل",
            "يتكامل",
            "مترابط",
            "توازن",
            "منظور",
            "اطار",
            "إطار",
            "يشمل",
            "معا",
            "معًا",
            "لا يفصل",
            "لا يفصل بين",
            "لا ينفصل",
            "لا ينفصل بين",
            "لا يفرق",
        ]
        strong_integration_markers = [
            "لا يفصل",
            "لا يفصل بين",
            "لا ينفصل",
            "لا ينفصل بين",
        ]
        conditional_markers = [
            "لا يتحقق",
            "لا يكتمل",
            "لا يتم",
            "متوقف على",
            "مشروط",
            "شرط",
            "لا يكون",
            "الا ب",
            "إلا ب",
            "لا بد",
            "ضرورة",
            "يتطلب",
        ]

        has_physical = "P004" in pillars

        sp_obj = MinedEdgeSpan(chunk_id=cid, span_start=int(sp.start), span_end=int(sp.end), quote=sent)
        pillar_set = set(pillars)

        reinforcement_markers = ["يدعم", "يعزز", "يساند", "يعضد"]

        if has_physical and _contains_any(sent_n, enabling_markers):
            for tgt in ["P001", "P002", "P003", "P005"]:
                if tgt in pillars:
                    out.append(
                        MinedEdge(
                            from_pillar_id="P004",
                            to_pillar_id=tgt,
                            relation_type="ENABLES",
                            spans=(sp_obj,),
                        )
                    )

        # Co-occurrence with "balance/system" cues across >=3 pillars -> pairwise COMPLEMENTS.
        # Reason: the framework often states "التوازن الروحي/المعرفي/..." as an integrated system.
        if len(pillars) >= 3 and _contains_any(sent_n, integration_markers):
            pairs = list(combinations(pillars[:5], 2))
            for a, b in pairs[:10]:
                if a == b:
                    continue
                out.append(MinedEdge(from_pillar_id=a, to_pillar_id=b, relation_type="COMPLEMENTS", spans=(sp_obj,)))
                out.append(MinedEdge(from_pillar_id=b, to_pillar_id=a, relation_type="COMPLEMENTS", spans=(sp_obj,)))

        # Integration language across 2+ pillars -> bidirectional COMPLEMENTS edges (pairwise).
        if _contains_any(sent_n, integration_markers) and len(pillars) >= 2:
            # Cap combinations for safety (avoid explosion on long lists).
            pairs = list(combinations(pillars[:5], 2))
            for a, b in pairs[:10]:
                if a == b:
                    continue
                out.append(MinedEdge(from_pillar_id=a, to_pillar_id=b, relation_type="COMPLEMENTS", spans=(sp_obj,)))
                out.append(MinedEdge(from_pillar_id=b, to_pillar_id=a, relation_type="COMPLEMENTS", spans=(sp_obj,)))

        # Explicit "no separation" integration -> also bidirectional REINFORCES (stronger than generic complements).
        if _contains_any(sent_n, strong_integration_markers) and len(pillars) >= 2:
            pairs = list(combinations(pillars[:5], 2))
            for a, b in pairs[:10]:
                if a == b:
                    continue
                out.append(MinedEdge(from_pillar_id=a, to_pillar_id=b, relation_type="REINFORCES", spans=(sp_obj,)))
                out.append(MinedEdge(from_pillar_id=b, to_pillar_id=a, relation_type="REINFORCES", spans=(sp_obj,)))

        # Generic integration cues (balance/system) can also ground a softer mutual reinforcement.
        if _contains_any(sent_n, integration_markers) and len(pillars) >= 2:
            pairs = list(combinations(pillars[:5], 2))
            for a, b in pairs[:10]:
                if a == b:
                    continue
                out.append(MinedEdge(from_pillar_id=a, to_pillar_id=b, relation_type="REINFORCES", spans=(sp_obj,)))
                out.append(MinedEdge(from_pillar_id=b, to_pillar_id=a, relation_type="REINFORCES", spans=(sp_obj,)))

        # Generic enabling/causal verb with multiple pillars -> directed ENABLES by mention order.
        if _contains_any(sent_n, enabling_markers) and len(pillars) >= 2:
            # If physical-specific already emitted, still allow other pair directions.
            for a, b in list(combinations(pillars[:5], 2))[:10]:
                d = _direct_by_order(sent_n, a, b)
                if not d:
                    continue
                fr, to = d
                if fr == to:
                    continue
                out.append(MinedEdge(from_pillar_id=fr, to_pillar_id=to, relation_type="ENABLES", spans=(sp_obj,)))

        # Softer reinforcement verbs -> directed REINFORCES by mention order.
        if _contains_any(sent_n, reinforcement_markers) and len(pillars) >= 2:
            for a, b in list(combinations(pillars[:5], 2))[:10]:
                d = _direct_by_order(sent_n, a, b)
                if not d:
                    continue
                fr, to = d
                if fr != to:
                    out.append(MinedEdge(from_pillar_id=fr, to_pillar_id=to, relation_type="REINFORCES", spans=(sp_obj,)))

        # Pattern group C: faith/heart reinforces physical (only when both mentioned)
        if ("P001" in pillars) and has_physical and _contains_any(sent_n, ["المؤمن", "قوته", "قوة", "قلبه", "ايمانه", "الإيمان"]):
            # Still require explicit reinforcement-like marker to be safe.
            if _contains_any(sent_n, ["يقوي", "تقوية", "قوة", "يعزز", "تعزيز"]):
                out.append(
                    MinedEdge(
                        from_pillar_id="P001",
                        to_pillar_id="P004",
                        relation_type="REINFORCES",
                        spans=(sp_obj,),
                    )
                )

        # Conditional claims: "لا يتحقق X إلا ب Y" / "متوقف على" -> X CONDITIONAL_ON Y by order around إلا.
        if _contains_any(sent_n, conditional_markers) and len(pillars) >= 2:
            # Heuristic: if "إلا" appears, treat pillar before "إلا" as dependent and after as condition.
            idx_illa = sent_n.find("الا")
            if idx_illa < 0:
                idx_illa = sent_n.find("إلا")
            before = sent_n[:idx_illa] if idx_illa > 0 else sent_n
            after = sent_n[idx_illa:] if idx_illa > 0 else sent_n
            before_p = [p for p in pillars if _first_index(before, p) >= 0]
            after_p = [p for p in pillars if _first_index(after, p) >= 0]
            if before_p and after_p:
                dep = before_p[0]
                cond = after_p[0]
                if dep != cond:
                    out.append(MinedEdge(from_pillar_id=dep, to_pillar_id=cond, relation_type="CONDITIONAL_ON", spans=(sp_obj,)))

        # Multi-span stitching: if we emitted something for this sentence, attach boundary spans from same chunk.
        # We do NOT create new edges from boundary-only sentences; we only add as extra evidence to existing edges.
        if out:
            # boundary spans are based on the whole chunk, but we only attach later during grouping.
            _ = pillar_set  # placeholder for clarity

    # Group by (from,to,rel) and stitch extra spans (boundary sentences) when available.
    grouped: dict[tuple[str, str, str], list[MinedEdgeSpan]] = {}
    for e in out:
        key = (e.from_pillar_id, e.to_pillar_id, e.relation_type)
        grouped.setdefault(key, []).extend(list(e.spans))

    # Add boundary spans as extra evidence for edges whose pillar pair is referenced in the chunk boundaries.
    boundaries = _boundary_sentences(text_ar=txt)
    # Boundaries returned with empty chunk_id; fill now.
    boundaries = [MinedEdgeSpan(chunk_id=cid, span_start=b.span_start, span_end=b.span_end, quote=b.quote) for b in boundaries]
    for (fr, to, rel), sp_list in list(grouped.items()):
        involved = {fr, to}
        added = 0
        added_generic = 0
        for b in boundaries:
            if added >= 2:
                break
            ps = set(_pillars_mentioned(b.quote))
            if ps and ps.intersection(involved):
                sp_list.append(b)
                added += 1
            elif (not ps) and added_generic < 1:
                # Generic boundary (no explicit pillar keywords). Attach at most one.
                sp_list.append(b)
                added += 1
                added_generic += 1
        grouped[(fr, to, rel)] = sp_list

    uniq: list[MinedEdge] = []
    for (fr, to, rel), sp_list in grouped.items():
        sp_tuple = _dedupe_spans(sp_list)[:6]
        if not sp_tuple:
            continue
        uniq.append(MinedEdge(from_pillar_id=fr, to_pillar_id=to, relation_type=rel, spans=sp_tuple))

    # Deterministic order for stability.
    uniq.sort(key=lambda e: (e.from_pillar_id, e.to_pillar_id, e.relation_type, e.spans[0].chunk_id, e.spans[0].span_start))
    return uniq


async def upsert_mined_edges(
    *,
    session: AsyncSession,
    mined: list[MinedEdge],
    created_by: str,
    strength_score: float = 0.8,
) -> dict[str, int]:
    """Insert mined edges + spans (idempotent).

    Returns counts for reporting.
    """

    inserted_edges = 0
    inserted_spans = 0

    for e in mined:
        # Insert edge (or reuse existing).
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
                      'pillar', :from_id, 'SCHOLAR_LINK', :relation_type,
                      'pillar', :to_id,
                      'rule_exact_match', :created_by, :justification,
                      :strength_score, 'approved'
                    )
                    ON CONFLICT DO NOTHING
                    RETURNING id
                    """
                ),
                {
                    "from_id": e.from_pillar_id,
                    "to_id": e.to_pillar_id,
                    "relation_type": e.relation_type,
                    "created_by": created_by,
                    "justification": "framework_mined",
                    "strength_score": float(min(0.95, float(strength_score) + 0.05 * max(0, (len(e.spans) - 1)))),
                },
            )
        ).fetchone()

        edge_id = str(row.id) if row and getattr(row, "id", None) else None
        if not edge_id:
            existing = (
                await session.execute(
                    text(
                        """
                        SELECT id::text AS id
                        FROM edge
                        WHERE from_type='pillar' AND from_id=:from_id
                          AND to_type='pillar' AND to_id=:to_id
                          AND rel_type='SCHOLAR_LINK'
                          AND relation_type=:relation_type
                        LIMIT 1
                        """
                    ),
                    {
                        "from_id": e.from_pillar_id,
                        "to_id": e.to_pillar_id,
                        "relation_type": e.relation_type,
                    },
                )
            ).fetchone()
            edge_id = str(existing.id) if existing and getattr(existing, "id", None) else None

        if not edge_id:
            # Hard fail closed: we must be able to attach spans to a real edge.
            raise RuntimeError("Failed to resolve edge_id for mined edge")

        if row and getattr(row, "id", None):
            inserted_edges += 1

        # Insert span grounding (hard gate): N spans allowed per edge.
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

    return {"inserted_edges": inserted_edges, "inserted_edge_spans": inserted_spans}

