"""Causal loop detection and reasoning for World Model.

This module provides:
- Deterministic graph cycle detection using DFS
- Loop classification (reinforcing/balancing) via polarity product
- Relevance ranking based on entity/pillar overlap
- On-the-fly summary generation from edge spans

Key algorithms:
- DFS-based cycle detection (Johnson's algorithm simplified)
- Polarity product: +1 = reinforcing, -1 = balancing
- Relevance score = (matched_nodes / total_nodes) * evidence_density
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from apps.api.core.world_model.schemas import (
    DetectedLoop,
    compute_loop_type,
)


@dataclass
class GraphNode:
    """Node in the mechanism graph."""
    id: str
    ref_kind: str
    ref_id: str
    label_ar: str


@dataclass
class GraphEdge:
    """Edge in the mechanism graph."""
    id: str
    from_node: str
    to_node: str
    relation_type: str
    polarity: int
    confidence: float
    spans: list[dict[str, Any]] = field(default_factory=list)


async def _load_edge_spans(
    session: AsyncSession,
    edge_ids: list[str],
) -> dict[str, list[dict[str, Any]]]:
    """Load evidence spans for a specific subset of edges.

    Reason:
    - Loop detection only needs topology + polarity, not spans.
    - Loading spans for *all* edges can be extremely expensive and can block the server.
    - We only fetch spans for the edges that actually appear in the selected loops.
    """
    if not edge_ids:
        return {}

    spans_result = await session.execute(
        text(
            """
            SELECT
                edge_id::text AS edge_id,
                chunk_id,
                span_start,
                span_end,
                quote
            FROM mechanism_edge_span
            WHERE edge_id::text = ANY(:edge_ids)
            ORDER BY edge_id, chunk_id, span_start
            """
        ),
        {"edge_ids": edge_ids},
    )

    spans_by_edge: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in spans_result.fetchall():
        spans_by_edge[str(row.edge_id)].append(
            {
                "chunk_id": str(row.chunk_id),
                "span_start": int(row.span_start),
                "span_end": int(row.span_end),
                "quote": str(row.quote),
            }
        )
    return spans_by_edge


async def load_mechanism_graph(
    session: AsyncSession,
    *,
    include_spans: bool = False,
) -> tuple[dict[str, GraphNode], list[GraphEdge]]:
    """Load mechanism nodes and edges from database.

    Args:
        include_spans: If True, load evidence spans for edges too.

    Returns:
        Tuple of (nodes_by_id, edges_list)
    """
    # Load nodes
    nodes_result = await session.execute(
        text("""
            SELECT id::text AS id, ref_kind, ref_id, label_ar
            FROM mechanism_node
            ORDER BY id
        """)
    )
    nodes_by_id: dict[str, GraphNode] = {}
    for row in nodes_result.fetchall():
        nodes_by_id[str(row.id)] = GraphNode(
            id=str(row.id),
            ref_kind=str(row.ref_kind),
            ref_id=str(row.ref_id),
            label_ar=str(row.label_ar),
        )
    
    # Load edges (spans are optionally loaded later)
    edges_result = await session.execute(
        text("""
            SELECT 
                e.id::text AS id,
                e.from_node::text AS from_node,
                e.to_node::text AS to_node,
                e.relation_type,
                e.polarity,
                e.confidence
            FROM mechanism_edge e
            ORDER BY e.id
        """)
    )
    
    edges: list[GraphEdge] = []
    edge_ids: list[str] = []
    for row in edges_result.fetchall():
        edge = GraphEdge(
            id=str(row.id),
            from_node=str(row.from_node),
            to_node=str(row.to_node),
            relation_type=str(row.relation_type),
            polarity=int(row.polarity),
            confidence=float(row.confidence or 0.5),
            spans=[],
        )
        edges.append(edge)
        edge_ids.append(edge.id)
    
    # Optional: load spans (avoid this on hot path unless explicitly requested).
    if include_spans and edge_ids:
        spans_by_edge = await _load_edge_spans(session, edge_ids=edge_ids)
        for edge in edges:
            edge.spans = spans_by_edge.get(edge.id, [])
    
    return nodes_by_id, edges


def _build_adjacency_list(edges: list[GraphEdge]) -> dict[str, list[tuple[str, GraphEdge]]]:
    """Build adjacency list from edges.
    
    Returns:
        Dict mapping node_id -> list of (neighbor_id, edge)
    """
    adj: dict[str, list[tuple[str, GraphEdge]]] = defaultdict(list)
    for edge in edges:
        adj[edge.from_node].append((edge.to_node, edge))
    return adj


def _find_cycles_dfs(
    adj: dict[str, list[tuple[str, GraphEdge]]],
    max_cycles: int = 50,
    max_cycle_length: int = 8,
) -> list[list[GraphEdge]]:
    """Find cycles in graph using DFS.
    
    Uses a simplified version of cycle detection:
    1. For each node, do DFS looking for back edges
    2. When we find a cycle, record the edge path
    3. Stop after max_cycles found
    
    Args:
        adj: Adjacency list
        max_cycles: Maximum cycles to find
        max_cycle_length: Maximum cycle length to consider
        
    Returns:
        List of cycles, each cycle is a list of edges
    """
    cycles: list[list[GraphEdge]] = []
    all_nodes = list(adj.keys())
    
    # Add nodes that only appear as targets
    for neighbors in adj.values():
        for neighbor_id, _ in neighbors:
            if neighbor_id not in all_nodes:
                all_nodes.append(neighbor_id)
    
    # For each starting node, try to find cycles
    for start in sorted(all_nodes):  # Sorted for determinism
        if len(cycles) >= max_cycles:
            break
        
        # DFS with path tracking
        stack: list[tuple[str, list[GraphEdge]]] = [(start, [])]
        visited_in_path: set[str] = set()
        
        while stack and len(cycles) < max_cycles:
            current, path = stack.pop()
            
            if len(path) > max_cycle_length:
                continue
            
            # Check if we've completed a cycle back to start
            if current == start and path:
                # Found a cycle! Make sure it's not a duplicate
                edge_ids = tuple(sorted(e.id for e in path))
                existing_ids = [tuple(sorted(e.id for e in c)) for c in cycles]
                if edge_ids not in existing_ids:
                    cycles.append(list(path))
                continue
            
            # Don't revisit nodes in current path (except for closing the cycle)
            if current in visited_in_path and current != start:
                continue
            
            visited_in_path.add(current)
            
            # Explore neighbors
            for neighbor_id, edge in adj.get(current, []):
                new_path = path + [edge]
                
                # If neighbor is start and we have a path, we might have a cycle
                if neighbor_id == start and new_path:
                    edge_ids = tuple(sorted(e.id for e in new_path))
                    existing_ids = [tuple(sorted(e.id for e in c)) for c in cycles]
                    if edge_ids not in existing_ids:
                        cycles.append(list(new_path))
                elif neighbor_id not in visited_in_path:
                    stack.append((neighbor_id, new_path))
            
            visited_in_path.discard(current)
    
    return cycles


def _cycle_to_detected_loop(
    cycle_edges: list[GraphEdge],
    nodes_by_id: dict[str, GraphNode],
) -> DetectedLoop:
    """Convert a cycle (list of edges) to a DetectedLoop."""
    
    # Extract ordered nodes
    node_ids: list[str] = []
    for edge in cycle_edges:
        if edge.from_node not in node_ids:
            node_ids.append(edge.from_node)
    
    # Get node labels
    nodes = [f"{nodes_by_id[nid].ref_kind}:{nodes_by_id[nid].ref_id}" 
             if nid in nodes_by_id else nid 
             for nid in node_ids]
    
    node_labels_ar = [nodes_by_id[nid].label_ar 
                      if nid in nodes_by_id else nid 
                      for nid in node_ids]
    
    # Get polarities
    polarities = [edge.polarity for edge in cycle_edges]
    
    # Compute loop type
    loop_type = compute_loop_type(polarities)
    
    # Collect all evidence spans
    evidence_spans: list[dict[str, Any]] = []
    for edge in cycle_edges:
        for span in edge.spans[:2]:  # Limit spans per edge
            evidence_spans.append(span)
    
    return DetectedLoop(
        loop_id=str(uuid4()),
        loop_type=loop_type,
        edge_ids=[edge.id for edge in cycle_edges],
        nodes=nodes,
        node_labels_ar=node_labels_ar,
        polarities=polarities,
        evidence_spans=evidence_spans,
    )


async def detect_loops(
    session: AsyncSession,
    max_loops: int = 20,
) -> list[DetectedLoop]:
    """Detect causal loops in mechanism graph.
    
    Algorithm:
    1. Build adjacency list from mechanism_edge
    2. DFS-based cycle detection
    3. For each cycle, compute polarity product
    4. Fetch evidence spans for all edges in cycle
    5. Return sorted by cycle length (shorter = more actionable)
    
    Args:
        session: Database session
        max_loops: Maximum number of loops to return
        
    Returns:
        List of detected loops, sorted by length
    """
    # Load graph topology only (no spans) for performance.
    nodes_by_id, edges = await load_mechanism_graph(session, include_spans=False)
    
    if not edges:
        return []
    
    # Build adjacency list
    adj = _build_adjacency_list(edges)
    
    # Find cycles
    cycles = _find_cycles_dfs(adj, max_cycles=max_loops * 2, max_cycle_length=8)
    
    # Fetch evidence spans only for edges that appear in selected cycles.
    selected_cycles = cycles[:max_loops]
    needed_edge_ids = sorted({e.id for cyc in selected_cycles for e in cyc})
    spans_by_edge = await _load_edge_spans(session, edge_ids=needed_edge_ids)

    # Attach spans for relevant edges (others remain empty).
    for edge in edges:
        if edge.id in spans_by_edge:
            edge.spans = spans_by_edge.get(edge.id, [])

    # Convert to DetectedLoop objects
    loops: list[DetectedLoop] = []
    for cycle_edges in selected_cycles:
        try:
            loop = _cycle_to_detected_loop(cycle_edges, nodes_by_id)
            loops.append(loop)
        except Exception:
            continue
    
    # Sort by length (shorter loops are more actionable)
    loops.sort(key=lambda l: len(l.edge_ids))
    
    return loops[:max_loops]


def compute_loop_relevance_score(
    loop: DetectedLoop,
    detected_entities: list[dict[str, Any]],
    detected_pillars: list[str],
) -> float:
    """Compute relevance score for a loop given question context.
    
    Score = (matched_nodes / total_nodes) * evidence_density
    
    Args:
        loop: The detected loop
        detected_entities: Entities detected in question
        detected_pillars: Pillar IDs detected in question
        
    Returns:
        Relevance score in [0, 1] range
    """
    if not loop.nodes:
        return 0.0
    
    # Build set of relevant identifiers
    relevant_ids: set[str] = set()
    
    for pillar in detected_pillars:
        relevant_ids.add(f"pillar:{pillar}")
        relevant_ids.add(pillar)
    
    for entity in detected_entities:
        etype = str(entity.get("entity_type") or entity.get("type") or "")
        eid = str(entity.get("entity_id") or entity.get("id") or "")
        if etype and eid:
            relevant_ids.add(f"{etype}:{eid}")
            relevant_ids.add(eid)
    
    # Count matched nodes
    matched = 0
    for node in loop.nodes:
        if node in relevant_ids:
            matched += 1
        # Also check just the ID part (after colon)
        if ":" in node:
            _, node_id = node.split(":", 1)
            if node_id in relevant_ids:
                matched += 1
    
    node_match_ratio = matched / len(loop.nodes) if loop.nodes else 0.0
    
    # Evidence density bonus
    evidence_count = len(loop.evidence_spans)
    evidence_bonus = min(0.3, evidence_count * 0.05)
    
    # Final score
    score = min(1.0, node_match_ratio + evidence_bonus)
    
    return round(score, 3)


def retrieve_relevant_loops(
    detected_loops: list[DetectedLoop],
    detected_entities: list[dict[str, Any]],
    detected_pillars: list[str],
    top_k: int = 5,
) -> list[DetectedLoop]:
    """Retrieve most relevant loops for a question.
    
    Ranks loops by relevance score and returns top K.
    
    Args:
        detected_loops: All detected loops
        detected_entities: Entities detected in question
        detected_pillars: Pillar IDs detected in question
        top_k: Number of loops to return
        
    Returns:
        Top K most relevant loops, sorted by relevance
    """
    if not detected_loops:
        return []
    
    # Score each loop
    scored: list[tuple[float, DetectedLoop]] = []
    for loop in detected_loops:
        score = compute_loop_relevance_score(loop, detected_entities, detected_pillars)
        scored.append((score, loop))
    
    # Sort by score descending, then by length ascending for ties
    scored.sort(key=lambda x: (-x[0], len(x[1].edge_ids)))
    
    # Return top K
    return [loop for _, loop in scored[:top_k]]


async def persist_detected_loops(
    session: AsyncSession,
    loops: list[DetectedLoop],
) -> int:
    """Persist detected loops to database.
    
    Args:
        session: Database session
        loops: Loops to persist
        
    Returns:
        Number of loops inserted
    """
    inserted = 0
    
    for loop in loops:
        if not loop.edge_ids:
            continue
        
        try:
            await session.execute(
                text("""
                    INSERT INTO feedback_loop (loop_id, edge_ids, loop_type)
                    VALUES (CAST(:loop_id AS uuid), CAST(:edge_ids AS uuid[]), :loop_type)
                    ON CONFLICT DO NOTHING
                """),
                {
                    "loop_id": loop.loop_id,
                    "edge_ids": list(loop.edge_ids),
                    "loop_type": loop.loop_type,
                }
            )
            inserted += 1
        except Exception:
            continue
    
    return inserted


async def load_persisted_loops(
    session: AsyncSession,
    max_loops: int = 20,
) -> list[DetectedLoop]:
    """Load persisted loops from the database (feedback_loop).

    Reason:
    - Runtime should not run full graph cycle detection; it is expensive and can hang.
    - Loops are mined offline and stored in `feedback_loop`.
    - We reconstruct DetectedLoop objects with labels + evidence spans for answer composition.
    """
    rows = (
        await session.execute(
            text(
                """
                SELECT
                    loop_id::text AS loop_id,
                    loop_type,
                    edge_ids::text[] AS edge_ids
                FROM feedback_loop
                ORDER BY loop_id
                LIMIT :lim
                """
            ),
            {"lim": int(max_loops)},
        )
    ).fetchall()
    if not rows:
        return []

    loop_specs: list[tuple[str, str, list[str]]] = []
    all_edge_ids: set[str] = set()
    for r in rows:
        eids = list(r.edge_ids or [])
        if not eids:
            continue
        loop_id = str(r.loop_id)
        loop_type = str(r.loop_type)
        loop_specs.append((loop_id, loop_type, eids))
        all_edge_ids.update([str(x) for x in eids])

    if not loop_specs:
        return []

    # Load edges referenced by loops
    edge_rows = (
        await session.execute(
            text(
                """
                SELECT
                    e.id::text AS id,
                    e.from_node::text AS from_node,
                    e.to_node::text AS to_node,
                    e.relation_type,
                    e.polarity,
                    e.confidence
                FROM mechanism_edge e
                WHERE e.id::text = ANY(:edge_ids)
                """
            ),
            {"edge_ids": sorted(list(all_edge_ids))},
        )
    ).fetchall()

    edges_by_id: dict[str, GraphEdge] = {}
    node_ids: set[str] = set()
    for row in edge_rows:
        eid = str(row.id)
        ge = GraphEdge(
            id=eid,
            from_node=str(row.from_node),
            to_node=str(row.to_node),
            relation_type=str(row.relation_type),
            polarity=int(row.polarity),
            confidence=float(row.confidence or 0.5),
            spans=[],
        )
        edges_by_id[eid] = ge
        node_ids.add(ge.from_node)
        node_ids.add(ge.to_node)

    # Load nodes referenced by these edges
    node_rows = (
        await session.execute(
            text(
                """
                SELECT id::text AS id, ref_kind, ref_id, label_ar
                FROM mechanism_node
                WHERE id::text = ANY(:node_ids)
                """
            ),
            {"node_ids": sorted(list(node_ids))},
        )
    ).fetchall()

    nodes_by_id: dict[str, GraphNode] = {}
    for row in node_rows:
        nid = str(row.id)
        nodes_by_id[nid] = GraphNode(
            id=nid,
            ref_kind=str(row.ref_kind),
            ref_id=str(row.ref_id),
            label_ar=str(row.label_ar),
        )

    # Load spans for the edges referenced by these loops (bounded set)
    spans_by_edge = await _load_edge_spans(session, edge_ids=sorted(list(all_edge_ids)))
    for eid, spans in spans_by_edge.items():
        if eid in edges_by_id:
            edges_by_id[eid].spans = spans

    loops: list[DetectedLoop] = []
    for loop_id, loop_type, eids in loop_specs:
        cycle_edges: list[GraphEdge] = []
        for eid in eids:
            if eid not in edges_by_id:
                cycle_edges = []
                break
            cycle_edges.append(edges_by_id[eid])
        if not cycle_edges:
            continue

        try:
            # Recompute loop_type deterministically if possible; otherwise trust stored.
            polarities = [e.polarity for e in cycle_edges]
            computed_type = compute_loop_type(polarities)
            final_type = computed_type if computed_type else str(loop_type)

            # Build node refs/labels in a stable order derived from edges.
            node_ids_order: list[str] = []
            for e in cycle_edges:
                if e.from_node not in node_ids_order:
                    node_ids_order.append(e.from_node)
                if e.to_node not in node_ids_order:
                    node_ids_order.append(e.to_node)

            nodes = [
                f"{nodes_by_id[nid].ref_kind}:{nodes_by_id[nid].ref_id}" if nid in nodes_by_id else nid
                for nid in node_ids_order
            ]
            node_labels_ar = [
                nodes_by_id[nid].label_ar if nid in nodes_by_id else nid
                for nid in node_ids_order
            ]

            evidence_spans: list[dict[str, Any]] = []
            for e in cycle_edges:
                for sp in (e.spans or [])[:2]:
                    evidence_spans.append(sp)

            loops.append(
                DetectedLoop(
                    loop_id=str(loop_id),
                    loop_type=str(final_type),
                    edge_ids=[e.id for e in cycle_edges],
                    nodes=nodes,
                    node_labels_ar=node_labels_ar,
                    polarities=polarities,
                    evidence_spans=evidence_spans,
                )
            )
        except Exception:
            continue

    # Deterministic ordering: shorter loops first (more actionable)
    loops.sort(key=lambda l: len(l.edge_ids))
    return loops[:max_loops]


async def get_loops_for_pillars(
    session: AsyncSession,
    pillar_ids: list[str],
    max_loops: int = 10,
) -> list[DetectedLoop]:
    """Get loops that involve specific pillars.
    
    Args:
        session: Database session
        pillar_ids: Pillar IDs to filter by
        max_loops: Maximum loops to return
        
    Returns:
        Loops involving the specified pillars
    """
    # Prefer persisted loops for runtime performance.
    all_loops = await load_persisted_loops(session, max_loops=max_loops * 2)
    
    if not pillar_ids:
        return all_loops[:max_loops]
    
    # Filter loops that involve any of the pillars
    relevant: list[DetectedLoop] = []
    for loop in all_loops:
        for node in loop.nodes:
            for pid in pillar_ids:
                if pid in node:
                    relevant.append(loop)
                    break
            else:
                continue
            break
    
    return relevant[:max_loops]


def generate_loop_summary_ar(loop: DetectedLoop) -> str:
    """Generate Arabic summary for a loop on-the-fly from edge spans.
    
    This implements adjustment #5: no stored summaries, generate from evidence.
    
    Args:
        loop: The detected loop
        
    Returns:
        Arabic summary string, or "غير منصوص عليه في الإطار" if no evidence
    """
    return loop.generate_summary_ar()
