"""Argument-grade graph store helpers.

This provides a minimal, grounded substrate for:
- Claim nodes
- EvidenceSpan nodes
- Argument links / conflicts / resolutions

Grounding rule:
- Any claim-to-evidence linkage MUST have a backing evidence span (chunk_id + offsets + quote).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession


@dataclass(frozen=True)
class ArgumentClaim:
    id: str
    text_ar: str
    entity_type: Optional[str]
    entity_id: Optional[str]


@dataclass(frozen=True)
class ArgumentEvidenceSpan:
    id: str
    chunk_id: str
    span_start: int
    span_end: int
    quote: str


async def create_claim(
    *,
    session: AsyncSession,
    text_ar: str,
    entity_type: str | None = None,
    entity_id: str | None = None,
) -> str:
    row = (
        await session.execute(
            text(
                """
                INSERT INTO argument_claim (text_ar, entity_type, entity_id)
                VALUES (:t, :et, :eid)
                RETURNING id::text
                """
            ),
            {"t": str(text_ar or "").strip(), "et": entity_type, "eid": entity_id},
        )
    ).fetchone()
    if not row:
        raise RuntimeError("Failed to create claim")
    return str(row[0])


async def create_evidence_span(
    *,
    session: AsyncSession,
    chunk_id: str,
    span_start: int,
    span_end: int,
    quote: str,
) -> str:
    row = (
        await session.execute(
            text(
                """
                INSERT INTO argument_evidence_span (chunk_id, span_start, span_end, quote)
                VALUES (:cid, :ss, :se, :q)
                RETURNING id::text
                """
            ),
            {
                "cid": str(chunk_id),
                "ss": int(span_start),
                "se": int(span_end),
                "q": str(quote or "").strip(),
            },
        )
    ).fetchone()
    if not row:
        raise RuntimeError("Failed to create evidence span")
    return str(row[0])


async def link_claim_supported_by_span(
    *,
    session: AsyncSession,
    claim_id: str,
    span_id: str,
    span_chunk_id: str,
    span_start: int,
    span_end: int,
    quote: str,
) -> str:
    """
    Create:
    - edge(claim --ARG_LINK/SUPPORTED_BY--> evidence_span)
    - edge_justification_span row (grounding)
    """

    edge_row = (
        await session.execute(
            text(
                """
                INSERT INTO edge (from_type, from_id, rel_type, relation_type, to_type, to_id, created_method, created_by, justification, status)
                VALUES ('claim', :cid, 'ARG_LINK', 'SUPPORTED_BY', 'evidence_span', :sid, 'rule_exact_match', 'argument_store', 'argument_span', 'approved')
                ON CONFLICT DO NOTHING
                RETURNING id::text
                """
            ),
            {"cid": str(claim_id), "sid": str(span_id)},
        )
    ).fetchone()
    # If already existed, fetch its id.
    if edge_row and edge_row[0]:
        edge_id = str(edge_row[0])
    else:
        row2 = (
            await session.execute(
                text(
                    """
                    SELECT id::text
                    FROM edge
                    WHERE from_type='claim' AND from_id=:cid
                      AND rel_type='ARG_LINK' AND relation_type='SUPPORTED_BY'
                      AND to_type='evidence_span' AND to_id=:sid
                    LIMIT 1
                    """
                ),
                {"cid": str(claim_id), "sid": str(span_id)},
            )
        ).fetchone()
        if not row2:
            raise RuntimeError("Failed to create/fetch edge for SUPPORTED_BY")
        edge_id = str(row2[0])

    # Grounding span must be present.
    if not (span_chunk_id and quote):
        raise ValueError("justification span must include chunk_id and quote")

    await session.execute(
        text(
            """
            INSERT INTO edge_justification_span (edge_id, chunk_id, span_start, span_end, quote)
            VALUES (:eid, :cid, :ss, :se, :q)
            ON CONFLICT DO NOTHING
            """
        ),
        {
            "eid": edge_id,
            "cid": str(span_chunk_id),
            "ss": int(span_start),
            "se": int(span_end),
            "q": str(quote),
        },
    )

    return edge_id


async def create_argument_link(
    *,
    session: AsyncSession,
    from_claim_id: str,
    to_claim_id: str,
    relation_type: str,
) -> str:
    """
    Create an argument link between claims (ENTAILS, TENSION_WITH, etc.).
    """

    row = (
        await session.execute(
            text(
                """
                INSERT INTO argument_argument (from_claim_id, to_claim_id, relation_type)
                VALUES (:f::uuid, :t::uuid, :rt)
                RETURNING id::text
                """
            ),
            {"f": str(from_claim_id), "t": str(to_claim_id), "rt": str(relation_type)},
        )
    ).fetchone()
    if not row:
        raise RuntimeError("Failed to create argument link")
    return str(row[0])

