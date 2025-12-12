"""
Graph Retriever Module

Retrieves evidence packets by traversing the knowledge graph edges.
"""

from typing import Any, Optional

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from apps.api.core.schemas import EntityType


async def get_entity_neighbors(
    session: AsyncSession,
    entity_type: EntityType,
    entity_id: str,
    relationship_types: Optional[list[str]] = None,
    direction: str = "both",
    status: str = "approved",
) -> list[dict[str, Any]]:
    """
    Get neighboring entities in the graph.

    Args:
        session: Database session.
        entity_type: Type of the source entity.
        entity_id: ID of the source entity.
        relationship_types: Filter for relationship types.
        direction: "outgoing", "incoming", or "both".
        status: Edge status filter (approved, candidate, all).

    Returns:
        List of neighboring entities with edge information.
    """
    neighbors = []

    # Build queries based on direction
    if direction in ("outgoing", "both"):
        query = """
            SELECT 
                e.id as edge_id,
                e.rel_type,
                e.to_type as neighbor_type,
                e.to_id as neighbor_id,
                e.confidence,
                e.created_method,
                e.score
            FROM edge e
            WHERE e.from_type = :entity_type
              AND e.from_id = :entity_id
        """

        params: dict[str, Any] = {
            "entity_type": entity_type.value,
            "entity_id": entity_id,
        }

        if relationship_types:
            query += " AND e.rel_type = ANY(:rel_types)"
            params["rel_types"] = relationship_types

        if status != "all":
            query += " AND e.status = :status"
            params["status"] = status

        result = await session.execute(text(query), params)
        for row in result.fetchall():
            neighbors.append({
                "edge_id": str(row.edge_id),
                "rel_type": row.rel_type,
                "neighbor_type": row.neighbor_type,
                "neighbor_id": row.neighbor_id,
                "direction": "outgoing",
                "confidence": row.confidence,
                "created_method": row.created_method,
                "score": row.score,
            })

    if direction in ("incoming", "both"):
        query = """
            SELECT 
                e.id as edge_id,
                e.rel_type,
                e.from_type as neighbor_type,
                e.from_id as neighbor_id,
                e.confidence,
                e.created_method,
                e.score
            FROM edge e
            WHERE e.to_type = :entity_type
              AND e.to_id = :entity_id
        """

        params = {
            "entity_type": entity_type.value,
            "entity_id": entity_id,
        }

        if relationship_types:
            query += " AND e.rel_type = ANY(:rel_types)"
            params["rel_types"] = relationship_types

        if status != "all":
            query += " AND e.status = :status"
            params["status"] = status

        result = await session.execute(text(query), params)
        for row in result.fetchall():
            neighbors.append({
                "edge_id": str(row.edge_id),
                "rel_type": row.rel_type,
                "neighbor_type": row.neighbor_type,
                "neighbor_id": row.neighbor_id,
                "direction": "incoming",
                "confidence": row.confidence,
                "created_method": row.created_method,
                "score": row.score,
            })

    return neighbors


async def expand_graph(
    session: AsyncSession,
    entity_type: EntityType,
    entity_id: str,
    depth: int = 2,
    relationship_types: Optional[list[str]] = None,
) -> list[dict[str, Any]]:
    """
    Expand the graph from an entity to a given depth.

    Args:
        session: Database session.
        entity_type: Type of the starting entity.
        entity_id: ID of the starting entity.
        depth: How many hops to traverse.
        relationship_types: Filter for relationship types.

    Returns:
        List of all entities reached.
    """
    visited: set[tuple[str, str]] = set()
    entities: list[dict[str, Any]] = []

    # Start with the given entity
    current_level = [(entity_type, entity_id)]
    visited.add((entity_type.value, entity_id))

    for level in range(depth):
        next_level = []

        for current_type, current_id in current_level:
            if isinstance(current_type, str):
                current_type = EntityType(current_type)

            neighbors = await get_entity_neighbors(
                session,
                current_type,
                current_id,
                relationship_types=relationship_types,
            )

            for neighbor in neighbors:
                key = (neighbor["neighbor_type"], neighbor["neighbor_id"])

                if key not in visited:
                    visited.add(key)
                    neighbor["depth"] = level + 1
                    entities.append(neighbor)
                    next_level.append((neighbor["neighbor_type"], neighbor["neighbor_id"]))

        current_level = next_level

        if not current_level:
            break

    return entities


async def get_hierarchy_path(
    session: AsyncSession,
    entity_type: EntityType,
    entity_id: str,
) -> list[dict[str, Any]]:
    """
    Get the hierarchy path from root to an entity.

    For sub-value: returns [pillar, core_value, sub_value]
    For core-value: returns [pillar, core_value]
    For pillar: returns [pillar]

    Args:
        session: Database session.
        entity_type: Type of the entity.
        entity_id: ID of the entity.

    Returns:
        List of entities in the path from root.
    """
    path = []

    if entity_type == EntityType.SUB_VALUE:
        # Get sub-value
        sv_result = await session.execute(
            text("SELECT id, name_ar, core_value_id FROM sub_value WHERE id = :id"),
            {"id": entity_id}
        )
        sv = sv_result.fetchone()

        if sv:
            # Get core value
            cv_result = await session.execute(
                text("SELECT id, name_ar, pillar_id FROM core_value WHERE id = :id"),
                {"id": sv.core_value_id}
            )
            cv = cv_result.fetchone()

            if cv:
                # Get pillar
                p_result = await session.execute(
                    text("SELECT id, name_ar FROM pillar WHERE id = :id"),
                    {"id": cv.pillar_id}
                )
                p = p_result.fetchone()

                if p:
                    path.append({
                        "entity_type": "pillar",
                        "entity_id": p.id,
                        "name_ar": p.name_ar,
                    })

                path.append({
                    "entity_type": "core_value",
                    "entity_id": cv.id,
                    "name_ar": cv.name_ar,
                })

            path.append({
                "entity_type": "sub_value",
                "entity_id": sv.id,
                "name_ar": sv.name_ar,
            })

    elif entity_type == EntityType.CORE_VALUE:
        # Get core value
        cv_result = await session.execute(
            text("SELECT id, name_ar, pillar_id FROM core_value WHERE id = :id"),
            {"id": entity_id}
        )
        cv = cv_result.fetchone()

        if cv:
            # Get pillar
            p_result = await session.execute(
                text("SELECT id, name_ar FROM pillar WHERE id = :id"),
                {"id": cv.pillar_id}
            )
            p = p_result.fetchone()

            if p:
                path.append({
                    "entity_type": "pillar",
                    "entity_id": p.id,
                    "name_ar": p.name_ar,
                })

            path.append({
                "entity_type": "core_value",
                "entity_id": cv.id,
                "name_ar": cv.name_ar,
            })

    elif entity_type == EntityType.PILLAR:
        # Get pillar
        p_result = await session.execute(
            text("SELECT id, name_ar FROM pillar WHERE id = :id"),
            {"id": entity_id}
        )
        p = p_result.fetchone()

        if p:
            path.append({
                "entity_type": "pillar",
                "entity_id": p.id,
                "name_ar": p.name_ar,
            })

    return path


async def get_related_values(
    session: AsyncSession,
    entity_type: EntityType,
    entity_id: str,
) -> list[dict[str, Any]]:
    """
    Get values related via RELATES_TO edges.

    This is for cross-pillar relationships.

    Args:
        session: Database session.
        entity_type: Type of entity.
        entity_id: Entity ID.

    Returns:
        List of related entities.
    """
    return await get_entity_neighbors(
        session,
        entity_type,
        entity_id,
        relationship_types=["RELATES_TO"],
        direction="both",
    )

