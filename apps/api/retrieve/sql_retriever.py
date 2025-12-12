"""
SQL Retriever Module

Retrieves evidence packets from PostgreSQL using exact entity lookups.
"""

from typing import Any, Optional

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from apps.api.core.schemas import EntityType, ChunkType


async def get_entity_chunks(
    session: AsyncSession,
    entity_type: EntityType,
    entity_id: str,
    chunk_types: Optional[list[ChunkType]] = None,
    limit: int = 20,
) -> list[dict[str, Any]]:
    """
    Get chunks for a specific entity.

    Args:
        session: Database session.
        entity_type: Type of entity.
        entity_id: Entity ID.
        chunk_types: Optional filter for chunk types.
        limit: Maximum chunks to return.

    Returns:
        List of evidence packets.
    """
    query = """
        SELECT 
            c.chunk_id,
            c.entity_type,
            c.entity_id,
            c.chunk_type,
            c.text_ar,
            c.source_doc_id,
            c.source_anchor
        FROM chunk c
        WHERE c.entity_type = :entity_type
          AND c.entity_id = :entity_id
    """

    params = {
        "entity_type": entity_type.value,
        "entity_id": entity_id,
    }

    if chunk_types:
        query += " AND c.chunk_type = ANY(:chunk_types)"
        params["chunk_types"] = [ct.value for ct in chunk_types]

    query += " ORDER BY c.created_at LIMIT :limit"
    params["limit"] = limit

    result = await session.execute(text(query), params)
    rows = result.fetchall()

    return [_row_to_evidence_packet(row) for row in rows]


async def get_chunks_with_refs(
    session: AsyncSession,
    entity_type: EntityType,
    entity_id: str,
    limit: int = 20,
) -> list[dict[str, Any]]:
    """
    Get chunks for an entity including reference information.

    Args:
        session: Database session.
        entity_type: Type of entity.
        entity_id: Entity ID.
        limit: Maximum chunks to return.

    Returns:
        List of evidence packets with refs.
    """
    # Get chunks
    chunks = await get_entity_chunks(session, entity_type, entity_id, limit=limit)

    # Get refs for each chunk
    for chunk in chunks:
        chunk["refs"] = await _get_chunk_refs(session, chunk["chunk_id"])

    return chunks


async def _get_chunk_refs(
    session: AsyncSession,
    chunk_id: str,
) -> list[dict[str, str]]:
    """Get references for a chunk."""
    result = await session.execute(
        text("""
            SELECT ref_type, ref
            FROM chunk_ref
            WHERE chunk_id = :chunk_id
        """),
        {"chunk_id": chunk_id}
    )

    return [{"type": row.ref_type, "ref": row.ref} for row in result.fetchall()]


async def get_pillar_hierarchy(
    session: AsyncSession,
    pillar_id: str,
) -> dict[str, Any]:
    """
    Get full hierarchy under a pillar.

    Args:
        session: Database session.
        pillar_id: Pillar ID.

    Returns:
        Dictionary with pillar and its core values and sub-values.
    """
    # Get pillar
    pillar_result = await session.execute(
        text("SELECT id, name_ar FROM pillar WHERE id = :id"),
        {"id": pillar_id}
    )
    pillar_row = pillar_result.fetchone()

    if not pillar_row:
        return {}

    # Get core values
    cv_result = await session.execute(
        text("""
            SELECT id, name_ar, definition_ar
            FROM core_value
            WHERE pillar_id = :pillar_id
            ORDER BY id
        """),
        {"pillar_id": pillar_id}
    )

    core_values = []
    for cv_row in cv_result.fetchall():
        # Get sub-values for this core value
        sv_result = await session.execute(
            text("""
                SELECT id, name_ar
                FROM sub_value
                WHERE core_value_id = :cv_id
                ORDER BY id
            """),
            {"cv_id": cv_row.id}
        )

        sub_values = [
            {"id": sv.id, "name_ar": sv.name_ar}
            for sv in sv_result.fetchall()
        ]

        core_values.append({
            "id": cv_row.id,
            "name_ar": cv_row.name_ar,
            "definition_ar": cv_row.definition_ar,
            "sub_values": sub_values,
        })

    return {
        "id": pillar_row.id,
        "name_ar": pillar_row.name_ar,
        "core_values": core_values,
    }


async def search_entities_by_name(
    session: AsyncSession,
    name_pattern: str,
    limit: int = 10,
) -> list[dict[str, Any]]:
    """
    Search entities by name pattern.

    Args:
        session: Database session.
        name_pattern: Pattern to search for.
        limit: Maximum results.

    Returns:
        List of matching entities.
    """
    results = []

    # Search pillars
    pillar_result = await session.execute(
        text("""
            SELECT 'pillar' as entity_type, id, name_ar
            FROM pillar
            WHERE name_ar ILIKE :pattern
            LIMIT :limit
        """),
        {"pattern": f"%{name_pattern}%", "limit": limit}
    )
    results.extend([dict(row._mapping) for row in pillar_result.fetchall()])

    # Search core values
    cv_result = await session.execute(
        text("""
            SELECT 'core_value' as entity_type, id, name_ar
            FROM core_value
            WHERE name_ar ILIKE :pattern
            LIMIT :limit
        """),
        {"pattern": f"%{name_pattern}%", "limit": limit}
    )
    results.extend([dict(row._mapping) for row in cv_result.fetchall()])

    # Search sub-values
    sv_result = await session.execute(
        text("""
            SELECT 'sub_value' as entity_type, id, name_ar
            FROM sub_value
            WHERE name_ar ILIKE :pattern
            LIMIT :limit
        """),
        {"pattern": f"%{name_pattern}%", "limit": limit}
    )
    results.extend([dict(row._mapping) for row in sv_result.fetchall()])

    return results[:limit]


def _row_to_evidence_packet(row) -> dict[str, Any]:
    """Convert a database row to evidence packet format."""
    return {
        "chunk_id": row.chunk_id,
        "entity_type": row.entity_type,
        "entity_id": row.entity_id,
        "chunk_type": row.chunk_type,
        "text_ar": row.text_ar,
        "source_doc_id": row.source_doc_id,
        "source_anchor": row.source_anchor,
        "refs": [],  # Filled in separately
    }

