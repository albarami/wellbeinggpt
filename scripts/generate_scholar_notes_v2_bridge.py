"""Generate a *bridge* scholar notes pack from existing grounded SCHOLAR_LINK edges.

Goal:
- Produce `data/scholar_notes/notes_v2_bridge.jsonl` whose rows are grounded by
  existing `edge_justification_span` rows.
- This does NOT invent new edges; it only reflects what already exists in DB.

Why:
- Stakeholder cross-pillar answers require grounded semantic edges.
- This pack helps surface existing bridges into retrieval as note chunks.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any, Optional

from sqlalchemy import text

from apps.api.core.database import get_session
from apps.api.ingest.scholar_notes_schema import ScholarRelationType
from apps.api.retrieve.normalize_ar import normalize_for_matching
from eval.datasets.source_loader import load_dotenv_if_present


def _relation_type_or_none(rt: str) -> Optional[str]:
    try:
        return str(ScholarRelationType(str(rt)))
    except Exception:
        return None


async def _fetch_grounded_cross_pillar_edges(*, session) -> list[dict[str, Any]]:
    # Reuse the same pillar-resolution logic as diagnostics, but include spans.
    res = await session.execute(
        text(
            """
            WITH edge_with_pillars AS (
              SELECT
                e.id::text AS edge_id,
                e.from_type, e.from_id,
                e.to_type, e.to_id,
                e.rel_type, e.relation_type,
                CASE
                  WHEN e.from_type = 'pillar' THEN e.from_id
                  WHEN e.from_type = 'core_value' THEN cv_from.pillar_id
                  WHEN e.from_type = 'sub_value' THEN cv_from2.pillar_id
                  ELSE NULL
                END AS from_pillar_id,
                CASE
                  WHEN e.to_type = 'pillar' THEN e.to_id
                  WHEN e.to_type = 'core_value' THEN cv_to.pillar_id
                  WHEN e.to_type = 'sub_value' THEN cv_to2.pillar_id
                  ELSE NULL
                END AS to_pillar_id
              FROM edge e
              LEFT JOIN core_value cv_from ON (e.from_type='core_value' AND cv_from.id = e.from_id)
              LEFT JOIN sub_value sv_from ON (e.from_type='sub_value' AND sv_from.id = e.from_id)
              LEFT JOIN core_value cv_from2 ON (sv_from.core_value_id = cv_from2.id)

              LEFT JOIN core_value cv_to ON (e.to_type='core_value' AND cv_to.id = e.to_id)
              LEFT JOIN sub_value sv_to ON (e.to_type='sub_value' AND sv_to.id = e.to_id)
              LEFT JOIN core_value cv_to2 ON (sv_to.core_value_id = cv_to2.id)
              WHERE e.rel_type = 'SCHOLAR_LINK' AND e.relation_type IS NOT NULL
            )
            SELECT
              ewp.edge_id,
              ewp.from_type, ewp.from_id,
              ewp.to_type, ewp.to_id,
              ewp.relation_type,
              ewp.from_pillar_id,
              ewp.to_pillar_id
            FROM edge_with_pillars ewp
            WHERE ewp.from_pillar_id IS NOT NULL
              AND ewp.to_pillar_id IS NOT NULL
              AND ewp.from_pillar_id <> ewp.to_pillar_id
              AND EXISTS (
                SELECT 1 FROM edge_justification_span js WHERE js.edge_id::text = ewp.edge_id
              )
            ORDER BY ewp.edge_id
            """
        )
    )
    return [dict(r._mapping) for r in res.fetchall()]


async def _fetch_edge_spans(*, session, edge_id: str) -> list[dict[str, Any]]:
    res = await session.execute(
        text(
            """
            SELECT
              js.chunk_id,
              js.span_start,
              js.span_end,
              js.quote,
              c.source_doc_id::text AS source_id
            FROM edge_justification_span js
            JOIN chunk c ON c.chunk_id = js.chunk_id
            WHERE js.edge_id::text = :eid
            ORDER BY js.chunk_id, js.span_start
            """
        ),
        {"eid": str(edge_id)},
    )
    return [dict(r._mapping) for r in res.fetchall()]


def _is_boundary_quote(q: str) -> bool:
    """
    Detect boundary/limits language inside a quote (deterministic).
    """

    t = normalize_for_matching(q or "")
    if not t:
        return False
    markers = [
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
        "تحذير",
        "تنبيه",
        "محاذير",
        "لا يتحقق",
        "لا يكتمل",
        "لا يتم",
        "متوقف على",
        "مشروط",
        "شرط",
        "لا بد",
        "ضرورة",
    ]
    return any(normalize_for_matching(m) in t for m in markers)


def _unique_quotes(spans: list[dict[str, Any]]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for sp in spans:
        q = str(sp.get("quote") or "").strip()
        if not q:
            continue
        qn = normalize_for_matching(q)
        if not qn or qn in seen:
            continue
        seen.add(qn)
        out.append(q)
    return out


def _note_row_from_edge(*, edge: dict[str, Any], spans: list[dict[str, Any]]) -> Optional[dict[str, Any]]:
    rt = _relation_type_or_none(str(edge.get("relation_type") or ""))
    if not rt:
        return None
    if not spans:
        return None

    from_pillar = str(edge.get("from_pillar_id") or "")
    to_pillar = str(edge.get("to_pillar_id") or "")
    if not (from_pillar and to_pillar):
        return None

    # Use first span quote as justification text (must contain quote verbatim for ingest gate).
    j = str(spans[0].get("quote") or "").strip()
    if not j:
        return None

    # Build "سبب الربط" and "حدود الربط" from spans only (no synthesis beyond quoting).
    boundary_spans = [sp for sp in spans if _is_boundary_quote(str(sp.get("quote") or ""))]
    core_spans = [sp for sp in spans if sp not in boundary_spans]

    core_quotes = _unique_quotes(core_spans)[:3] or _unique_quotes(spans)[:3]
    boundary_quotes = _unique_quotes(boundary_spans)[:2]

    deep_lines: list[str] = []
    deep_lines.append("سبب الربط (من النص):")
    for q in core_quotes:
        deep_lines.append(f"- {q}")
    if not core_quotes:
        deep_lines.append("- غير منصوص عليه في الإطار")

    deep_lines.append("")
    if boundary_quotes:
        deep_lines.append("حدود الربط (من النص):")
        for q in boundary_quotes:
            deep_lines.append(f"- {q}")
    else:
        deep_lines.append("حدود الربط: غير منصوص عليه في الإطار.")

    ev_spans = []
    for sp in spans[:6]:
        ev_spans.append(
            {
                "source_id": str(sp.get("source_id") or ""),
                "chunk_id": str(sp.get("chunk_id") or ""),
                "span_start": int(sp.get("span_start") or 0),
                "span_end": int(sp.get("span_end") or 0),
                "quote": str(sp.get("quote") or ""),
            }
        )

    note_id = f"bridge_{str(edge.get('edge_id') or '')[:12]}"
    return {
        "note_id": note_id,
        "pillar_id": from_pillar,
        "core_value_id": None,
        "sub_value_id": None,
        "title_ar": f"جسر ربط: {from_pillar} → {to_pillar} ({rt})",
        "definition_ar": "",
        "deep_explanation_ar": "\n".join([x for x in deep_lines if x]).strip(),
        "cross_pillar_links": [
            {
                "target_pillar_id": to_pillar,
                "target_sub_value_id": None,
                "relation_type": rt,
                "justification_ar": j,
            }
        ],
        "applied_scenarios": [],
        "common_misunderstandings": [],
        "evidence_spans": ev_spans,
        "tags": (["bridge", "v2"] + (["boundaries_na"] if not boundary_quotes else [])),
        "version": "v2",
    }


async def _run(*, out_path: Path) -> int:
    load_dotenv_if_present()
    rows_out: list[dict[str, Any]] = []
    async with get_session() as session:
        edges = await _fetch_grounded_cross_pillar_edges(session=session)
        for e in edges:
            spans = await _fetch_edge_spans(session=session, edge_id=str(e.get("edge_id") or ""))
            row = _note_row_from_edge(edge=e, spans=spans)
            if row is not None:
                rows_out.append(row)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("", encoding="utf-8")
    with out_path.open("w", encoding="utf-8") as f:
        for r in rows_out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Wrote {len(rows_out)} bridge notes -> {out_path.as_posix()}")
    return len(rows_out)


def main() -> None:
    asyncio.run(_run(out_path=Path("data/scholar_notes/notes_v2_bridge.jsonl")))


if __name__ == "__main__":
    main()

