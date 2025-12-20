"""Database upsert helpers for mechanism mining."""

from __future__ import annotations

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from apps.api.core.world_model.schemas import compute_edge_confidence

from apps.api.graph.mechanism_miner_types import MinedMechanismEdge


async def get_or_create_mechanism_node(
    session: AsyncSession,
    ref_kind: str,
    ref_id: str,
    label_ar: str,
    source_id: str | None = None,
) -> str:
    """Get or create a mechanism node and return its UUID."""
    result = await session.execute(
        text(
            """
            SELECT id::text AS id FROM mechanism_node
            WHERE ref_kind = :ref_kind AND ref_id = :ref_id
            LIMIT 1
            """
        ),
        {"ref_kind": ref_kind, "ref_id": ref_id},
    )
    row = result.fetchone()
    if row:
        return str(row.id)

    result = await session.execute(
        text(
            """
            INSERT INTO mechanism_node (ref_kind, ref_id, label_ar, source_id)
            VALUES (:ref_kind, :ref_id, :label_ar, CAST(:source_id AS uuid))
            ON CONFLICT (ref_kind, ref_id) DO UPDATE SET label_ar = EXCLUDED.label_ar
            RETURNING id::text AS id
            """
        ),
        {
            "ref_kind": ref_kind,
            "ref_id": ref_id,
            "label_ar": label_ar,
            "source_id": source_id,
        },
    )
    row = result.fetchone()
    return str(row.id) if row else ""


async def upsert_mechanism_edges(
    *,
    session: AsyncSession,
    mined: list[MinedMechanismEdge],
    source_id: str | None = None,
    pillar_labels: dict[str, str] | None = None,
) -> dict[str, int]:
    """Insert mined mechanism edges + spans (idempotent)."""
    inserted_edges = 0
    inserted_spans = 0

    labels = pillar_labels or {
        "P001": "الركيزة الروحية",
        "P002": "الركيزة العاطفية",
        "P003": "الركيزة الفكرية",
        "P004": "الركيزة البدنية",
        "P005": "الركيزة الاجتماعية",
    }

    for e in mined:
        if not e.spans:
            continue  # Hard gate

        from_label = labels.get(e.from_ref_id, e.from_ref_id)
        from_node_id = await get_or_create_mechanism_node(
            session, e.from_ref_kind, e.from_ref_id, from_label, source_id
        )

        to_label = labels.get(e.to_ref_id, e.to_ref_id)
        to_node_id = await get_or_create_mechanism_node(
            session, e.to_ref_kind, e.to_ref_id, to_label, source_id
        )

        if not from_node_id or not to_node_id:
            continue

        span_count = len(e.spans)
        chunk_diversity = len(set(sp.chunk_id for sp in e.spans))
        confidence = compute_edge_confidence(span_count, chunk_diversity, span_count == 1)

        # Idempotency without relying on a DB unique index:
        # Check if the semantic edge already exists; if yes, reuse it.
        existing = await session.execute(
            text(
                """
                SELECT id::text AS id FROM mechanism_edge
                WHERE from_node = CAST(:from_node AS uuid)
                  AND to_node = CAST(:to_node AS uuid)
                  AND relation_type = :relation_type
                LIMIT 1
                """
            ),
            {
                "from_node": from_node_id,
                "to_node": to_node_id,
                "relation_type": e.relation_type,
            },
        )
        ex_row = existing.fetchone()
        edge_id = str(ex_row.id) if ex_row else None

        if not edge_id:
            result = await session.execute(
                text(
                    """
                    INSERT INTO mechanism_edge (from_node, to_node, relation_type, polarity, confidence)
                    VALUES (CAST(:from_node AS uuid), CAST(:to_node AS uuid), :relation_type, :polarity, :confidence)
                    RETURNING id::text AS id
                    """
                ),
                {
                    "from_node": from_node_id,
                    "to_node": to_node_id,
                    "relation_type": e.relation_type,
                    "polarity": e.polarity,
                    "confidence": confidence,
                },
            )
            row = result.fetchone()
            edge_id = str(row.id) if row else None
            if edge_id:
                inserted_edges += 1

        if not edge_id:
            continue

        for sp in list(e.spans)[:8]:
            if not sp.quote.strip():
                continue
            span_res = await session.execute(
                text(
                    """
                    INSERT INTO mechanism_edge_span (edge_id, chunk_id, span_start, span_end, quote)
                    VALUES (CAST(:edge_id AS uuid), :chunk_id, :span_start, :span_end, :quote)
                    ON CONFLICT DO NOTHING
                    RETURNING id::text AS id
                    """
                ),
                {
                    "edge_id": edge_id,
                    "chunk_id": sp.chunk_id,
                    "span_start": sp.span_start,
                    "span_end": sp.span_end,
                    "quote": sp.quote,
                },
            )
            if span_res.fetchone():
                inserted_spans += 1

    return {"inserted_edges": inserted_edges, "inserted_spans": inserted_spans}

