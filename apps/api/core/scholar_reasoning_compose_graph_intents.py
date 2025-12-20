"""Deterministic graph-intent composers (evidence-only).

Handles:
- cross_pillar path (step-by-step)
- network (>= N links across pillars)
- tension/reconciliation (tension -> reconciliation -> boundaries)
"""

from __future__ import annotations

from typing import Any

from apps.api.core.schemas import Citation


def _cite_span(citations: list[Citation], span: dict[str, Any]) -> None:
    cid = str(span.get("chunk_id") or "").strip()
    if not cid:
        return
    citations.append(Citation(chunk_id=cid, source_anchor="", ref=None))


def _edge_used_record(ed: dict[str, Any], picked_spans: list[dict[str, Any]]) -> dict[str, Any] | None:
    edge_id = str(ed.get("edge_id") or "").strip()
    rt = str(ed.get("relation_type") or "").strip()
    n_type = str(ed.get("neighbor_type") or "").strip()
    n_id = str(ed.get("neighbor_id") or "").strip()
    src_type = str(ed.get("source_type") or "").strip()
    src_id = str(ed.get("source_id") or "").strip()
    direction = str(ed.get("direction") or "").strip()
    if not (edge_id and rt and n_type and n_id and src_type and src_id and picked_spans):
        return None
    from_node = f"{src_type}:{src_id}"
    to_node = f"{n_type}:{n_id}"
    if direction == "incoming":
        from_node, to_node = to_node, from_node
    return {
        "edge_id": edge_id,
        "from_node": from_node,
        "to_node": to_node,
        "relation_type": rt,
        "justification_spans": picked_spans,
    }


def compose_cross_pillar_path_answer(
    *,
    packets: list[dict[str, Any]],
    semantic_edges: list[dict[str, Any]],
    max_steps: int = 3,
    prefer_more_claims: bool = False,
) -> tuple[str, list[Citation], list[dict[str, Any]]]:
    """
    Compose a step-by-step cross-pillar path using semantic edges.

    Evidence-only: each step cites a justification span quote.
    """

    citations: list[Citation] = []
    used_edges: list[dict[str, Any]] = []

    parts: list[str] = []
    # Add lightweight grounded context before the path.
    # Prefer pillar-level definitions/evidence for the pillars involved in the path.
    pillar_ids: set[str] = set()
    for ed in (semantic_edges or [])[:8]:
        if str(ed.get("source_type") or "") == "pillar":
            pillar_ids.add(str(ed.get("source_id") or ""))
        if str(ed.get("neighbor_type") or "") == "pillar":
            pillar_ids.add(str(ed.get("neighbor_id") or ""))

    defs = [p for p in (packets or []) if p.get("chunk_type") == "definition"]
    evs = [p for p in (packets or []) if p.get("chunk_type") == "evidence"]
    if pillar_ids:
        defs = [p for p in defs if str(p.get("entity_type") or "") == "pillar" and str(p.get("entity_id") or "") in pillar_ids] or defs
        evs = [p for p in evs if str(p.get("entity_type") or "") == "pillar" and str(p.get("entity_id") or "") in pillar_ids] or evs

    parts.append("تعريف المفهوم داخل الإطار")
    added = 0
    for p in defs[:2]:
        t = str(p.get("text_ar") or "").strip()
        if not t:
            continue
        parts.append(f"- {t}")
        cid = str(p.get("chunk_id") or "")
        if cid:
            citations.append(Citation(chunk_id=cid, source_anchor=str(p.get("source_anchor") or ""), ref=None))
        added += 1
    if added == 0:
        parts.append("- غير منصوص عليه")

    parts.append("")
    parts.append("التأصيل والأدلة (مختصر ومركز)")
    added = 0
    for p in evs[:2]:
        t = str(p.get("text_ar") or "").strip()
        if not t:
            continue
        parts.append(f"- {t}")
        cid = str(p.get("chunk_id") or "")
        if cid:
            citations.append(Citation(chunk_id=cid, source_anchor=str(p.get("source_anchor") or ""), ref=None))
        added += 1
    if added == 0:
        parts.append("- غير منصوص عليه")

    parts.append("")
    parts.append("الربط بين الركائز (مع سبب الربط)")

    steps = 0
    for ed in (semantic_edges or []):
        if steps >= max_steps:
            break
        spans = list(ed.get("justification_spans") or [])
        if not spans:
            continue
        rt = str(ed.get("relation_type") or "").strip()
        n_type = str(ed.get("neighbor_type") or "").strip()
        n_id = str(ed.get("neighbor_id") or "").strip()
        src_type = str(ed.get("source_type") or "").strip()
        src_id = str(ed.get("source_id") or "").strip()
        if not (rt and n_type and n_id and src_type and src_id):
            continue

        direction = str(ed.get("direction") or "").strip()
        from_node = f"{src_type}:{src_id}"
        to_node = f"{n_type}:{n_id}"
        if direction == "incoming":
            from_node, to_node = to_node, from_node

        pick = spans[: (2 if prefer_more_claims else 1)]
        pick_out: list[dict[str, Any]] = []
        for sp in pick:
            quote = str(sp.get("quote") or "").strip()
            if not quote:
                continue
            steps += 1
            parts.append(f"- خطوة {steps}: ({rt}) {from_node} → {to_node} — شاهد: {quote}")
            _cite_span(citations, sp)
            pick_out.append(
                {
                    "chunk_id": str(sp.get("chunk_id") or ""),
                    "span_start": int(sp.get("span_start") or 0),
                    "span_end": int(sp.get("span_end") or 0),
                    "quote": quote,
                }
            )
            if steps >= max_steps:
                break
        rec = _edge_used_record(ed, pick_out)
        if rec:
            used_edges.append(rec)

    if steps == 0:
        parts.append("- (لا توجد روابط دلالية مُبرَّرة ضمن الأدلة المسترجعة)")

    parts.append("")
    parts.append("خلاصة تنفيذية (3 نقاط)")
    # Evidence-only summary: recycle packet texts.
    added = 0
    for p in (defs[:2] + evs[:3] + list(packets or [])[:6])[:12]:
        if added >= 3:
            break
        t = str(p.get("text_ar") or "").strip()
        if not t:
            continue
        parts.append(f"- {t}")
        cid = str(p.get("chunk_id") or "")
        if cid:
            citations.append(Citation(chunk_id=cid, source_anchor=str(p.get("source_anchor") or ""), ref=None))
        added += 1
    while added < 3:
        parts.append("- غير منصوص عليه")
        added += 1

    return "\n".join(parts).strip(), citations, used_edges


def _edge_priority(ed: dict[str, Any]) -> int:
    """
    Prioritize value-level edges over pillar-level edges.
    
    Reason: "اختر قيمة محورية" questions need value-level central nodes,
    not just pillar → pillar links. This is a real sophistication jump.
    
    Priority (lower = better):
    - 0: core_value ↔ core_value (cross-pillar)
    - 1: sub_value ↔ sub_value (cross-pillar)
    - 2: core_value ↔ pillar or sub_value ↔ pillar
    - 3: pillar ↔ pillar
    """
    src_type = str(ed.get("source_type") or "").lower()
    n_type = str(ed.get("neighbor_type") or "").lower()
    
    # Both are values (best)
    if src_type in ("core_value", "sub_value") and n_type in ("core_value", "sub_value"):
        if src_type == "core_value" and n_type == "core_value":
            return 0
        return 1
    
    # One is value, one is pillar
    if src_type in ("core_value", "sub_value") or n_type in ("core_value", "sub_value"):
        return 2
    
    # Both pillars (default/fallback)
    return 3


def compose_network_answer(
    *,
    packets: list[dict[str, Any]],
    semantic_edges: list[dict[str, Any]],
    max_links: int = 5,
) -> tuple[str, list[Citation], list[dict[str, Any]]]:
    """
    Compose a network answer: multiple links with relation_type + per-link evidence.
    
    UPGRADE: Prefers value-level central nodes and value-level cross-pillar links.
    Falls back to pillar-level edges only if value-level edges don't exist.
    """

    citations: list[Citation] = []
    used_edges: list[dict[str, Any]] = []
    parts: list[str] = []

    # UPGRADE: Sort edges by priority (value-level first, pillar-level last)
    sorted_edges = sorted(semantic_edges or [], key=_edge_priority)
    
    # Track if we found value-level edges
    has_value_edges = any(_edge_priority(e) < 3 for e in sorted_edges[:max_links * 2])

    parts.append("الربط بين الركائز (مع سبب الربط)")
    if has_value_edges:
        parts.append("- (شبكة على مستوى القيم المحورية)")
    
    links = 0
    seen_pairs: set[tuple[str, str, str]] = set()
    for ed in sorted_edges[: (max_links * 2)]:
        if links >= max_links:
            break
        spans = list(ed.get("justification_spans") or [])
        if not spans:
            continue
        rec = _edge_used_record(ed, [spans[0]])
        if not rec:
            continue
        key = (rec["from_node"], rec["to_node"], rec["relation_type"])
        if key in seen_pairs:
            continue
        seen_pairs.add(key)
        quote = str(spans[0].get("quote") or "").strip()
        if not quote:
            continue
        parts.append(f"- ({rec['relation_type']}) {rec['from_node']} → {rec['to_node']} — شاهد: {quote}")
        _cite_span(citations, spans[0])
        rec["justification_spans"] = [
            {
                "chunk_id": str(spans[0].get("chunk_id") or ""),
                "span_start": int(spans[0].get("span_start") or 0),
                "span_end": int(spans[0].get("span_end") or 0),
                "quote": quote,
            }
        ]
        used_edges.append(rec)
        links += 1

    if links == 0:
        parts.append("- (لا توجد روابط دلالية مُبرَّرة ضمن الأدلة المسترجعة)")

    parts.append("")
    parts.append("خلاصة تنفيذية (3 نقاط)")
    added = 0
    for p in (packets or [])[:8]:
        if added >= 3:
            break
        t = str(p.get("text_ar") or "").strip()
        if not t:
            continue
        parts.append(f"- {t}")
        cid = str(p.get("chunk_id") or "")
        if cid:
            citations.append(Citation(chunk_id=cid, source_anchor=str(p.get("source_anchor") or ""), ref=None))
        added += 1
    while added < 3:
        parts.append("- غير منصوص عليه")
        added += 1

    return "\n".join(parts).strip(), citations, used_edges


def compose_tension_answer(
    *,
    packets: list[dict[str, Any]],
    semantic_edges: list[dict[str, Any]],
) -> tuple[str, list[Citation], list[dict[str, Any]]]:
    """
    Compose a tension/reconciliation answer if TENSION_WITH / RESOLVES_WITH edges exist.
    """

    citations: list[Citation] = []
    used_edges: list[dict[str, Any]] = []
    parts: list[str] = []

    parts.append("الربط بين الركائز (مع سبب الربط)")

    picked = [e for e in (semantic_edges or []) if str(e.get("relation_type") or "") in {"TENSION_WITH", "RESOLVES_WITH"}]
    if not picked:
        parts.append("- (لا توجد روابط (تعارض/توفيق) مُبرَّرة ضمن الأدلة المسترجعة)")
    else:
        for ed in picked[:2]:
            spans = list(ed.get("justification_spans") or [])
            if not spans:
                continue
            quote = str(spans[0].get("quote") or "").strip()
            if not quote:
                continue
            rec = _edge_used_record(ed, [spans[0]])
            if not rec:
                continue
            parts.append(f"- ({rec['relation_type']}) {rec['from_node']} ↔ {rec['to_node']} — شاهد: {quote}")
            _cite_span(citations, spans[0])
            rec["justification_spans"] = [
                {
                    "chunk_id": str(spans[0].get("chunk_id") or ""),
                    "span_start": int(spans[0].get("span_start") or 0),
                    "span_end": int(spans[0].get("span_end") or 0),
                    "quote": quote,
                }
            ]
            used_edges.append(rec)

    parts.append("")
    parts.append("خلاصة تنفيذية (3 نقاط)")
    added = 0
    for p in (packets or [])[:8]:
        if added >= 3:
            break
        t = str(p.get("text_ar") or "").strip()
        if not t:
            continue
        parts.append(f"- {t}")
        cid = str(p.get("chunk_id") or "")
        if cid:
            citations.append(Citation(chunk_id=cid, source_anchor=str(p.get("source_anchor") or ""), ref=None))
        added += 1
    while added < 3:
        parts.append("- غير منصوص عليه")
        added += 1

    return "\n".join(parts).strip(), citations, used_edges

