"""Ingestion loader: deterministic edge building.

Reason: keep each module <500 LOC.
"""

from __future__ import annotations

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession


async def build_edges_for_source(session: AsyncSession, source_doc_id: str) -> None:
    """Build minimal graph edges in Postgres for a given source_doc_id."""
    await session.execute(
        text(
            """
            INSERT INTO edge (from_type, from_id, rel_type, to_type, to_id,
                              created_method, created_by, justification, status)
            SELECT 'pillar', cv.pillar_id, 'CONTAINS', 'core_value', cv.id,
                   'rule_exact_match', 'system', 'hierarchy', 'approved'
            FROM core_value cv
            WHERE cv.source_doc_id = :source_doc_id
            ON CONFLICT DO NOTHING
            """
        ),
        {"source_doc_id": source_doc_id},
    )

    await session.execute(
        text(
            """
            INSERT INTO edge (from_type, from_id, rel_type, to_type, to_id,
                              created_method, created_by, justification, status)
            SELECT 'core_value', sv.core_value_id, 'CONTAINS', 'sub_value', sv.id,
                   'rule_exact_match', 'system', 'hierarchy', 'approved'
            FROM sub_value sv
            WHERE sv.source_doc_id = :source_doc_id
            ON CONFLICT DO NOTHING
            """
        ),
        {"source_doc_id": source_doc_id},
    )

    await session.execute(
        text(
            """
            INSERT INTO edge (from_type, from_id, rel_type, to_type, to_id,
                              created_method, created_by, justification, status)
            SELECT e.entity_type, e.entity_id, 'SUPPORTED_BY', 'evidence', e.id::text,
                   'rule_exact_match', 'system', 'evidence_link', 'approved'
            FROM evidence e
            WHERE e.source_doc_id = :source_doc_id
            ON CONFLICT DO NOTHING
            """
        ),
        {"source_doc_id": source_doc_id},
    )

    await session.execute(
        text(
            """
            INSERT INTO edge (from_type, from_id, rel_type, to_type, to_id,
                              created_method, created_by, justification, status)
            SELECT
                'evidence',
                e.id::text,
                'REFERS_TO',
                'ref',
                e.evidence_type || ':' || e.ref_norm,
                'rule_exact_match',
                'system',
                e.evidence_type || ':' || e.ref_norm,
                'approved'
            FROM evidence e
            WHERE e.source_doc_id = :source_doc_id
              AND e.ref_norm IS NOT NULL
              AND e.ref_norm <> ''
            ON CONFLICT DO NOTHING
            """
        ),
        {"source_doc_id": source_doc_id},
    )

    await session.execute(
        text(
            """
            INSERT INTO edge (from_type, from_id, rel_type, to_type, to_id,
                              created_method, created_by, justification, status)
            SELECT
                e.entity_type,
                e.entity_id,
                'MENTIONS_REF',
                'ref',
                e.evidence_type || ':' || e.ref_norm,
                'rule_exact_match',
                'system',
                e.evidence_type || ':' || e.ref_norm,
                'approved'
            FROM evidence e
            WHERE e.source_doc_id = :source_doc_id
              AND e.ref_norm IS NOT NULL
              AND e.ref_norm <> ''
            ON CONFLICT DO NOTHING
            """
        ),
        {"source_doc_id": source_doc_id},
    )

    await session.execute(
        text(
            """
            WITH refs AS (
                SELECT DISTINCT
                    e.entity_type,
                    e.entity_id,
                    e.ref_norm,
                    e.evidence_type
                FROM evidence e
                WHERE e.source_doc_id = :source_doc_id
                  AND e.ref_norm IS NOT NULL
                  AND e.ref_norm <> ''
            ),
            pairs AS (
                SELECT
                    r1.entity_type AS from_type,
                    r1.entity_id AS from_id,
                    r2.entity_type AS to_type,
                    r2.entity_id AS to_id,
                    r1.ref_norm AS ref_norm,
                    r1.evidence_type AS evidence_type
                FROM refs r1
                JOIN refs r2
                  ON r1.ref_norm = r2.ref_norm
                 AND (r1.entity_type, r1.entity_id) < (r2.entity_type, r2.entity_id)
            )
            INSERT INTO edge (from_type, from_id, rel_type, to_type, to_id,
                              created_method, created_by, justification, status)
            SELECT
                p.from_type,
                p.from_id,
                'SHARES_REF',
                p.to_type,
                p.to_id,
                'rule_exact_match',
                'system',
                p.evidence_type || ':' || p.ref_norm,
                'approved'
            FROM pairs p
            ON CONFLICT DO NOTHING
            """
        ),
        {"source_doc_id": source_doc_id},
    )

    await session.execute(
        text(
            """
            WITH dups AS (
                SELECT name_ar
                FROM sub_value
                WHERE source_doc_id = :source_doc_id
                GROUP BY name_ar
                HAVING count(*) > 1
            ),
            pairs AS (
                SELECT
                    sv1.id AS from_id,
                    sv2.id AS to_id,
                    sv1.name_ar AS name_ar
                FROM sub_value sv1
                JOIN sub_value sv2
                  ON sv1.name_ar = sv2.name_ar
                 AND sv1.core_value_id <> sv2.core_value_id
                 AND sv1.id < sv2.id
                JOIN dups d ON d.name_ar = sv1.name_ar
                WHERE sv1.source_doc_id = :source_doc_id
                  AND sv2.source_doc_id = :source_doc_id
            )
            INSERT INTO edge (from_type, from_id, rel_type, to_type, to_id,
                              created_method, created_by, justification, status)
            SELECT
                'sub_value',
                p.from_id,
                'SAME_NAME',
                'sub_value',
                p.to_id,
                'rule_exact_match',
                'system',
                p.name_ar,
                'approved'
            FROM pairs p
            ON CONFLICT DO NOTHING
            """
        ),
        {"source_doc_id": source_doc_id},
    )

