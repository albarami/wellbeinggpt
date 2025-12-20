"""Caching layer for World Model components.

This module provides:
- In-memory caching for loop detection results
- Mechanism graph stats caching
- TTL-based cache invalidation
- Cache invalidation on edge insert

Target: p95 < 10s for deep synthesis.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession


@dataclass
class CacheEntry:
    """A cache entry with TTL."""
    
    value: Any
    created_at: float
    ttl_seconds: float
    
    def is_expired(self) -> bool:
        """Check if this entry has expired."""
        return (time.time() - self.created_at) > self.ttl_seconds


@dataclass
class MechanismGraphStats:
    """Cached statistics about the mechanism graph."""
    
    total_nodes: int = 0
    total_edges: int = 0
    edges_by_pillar: dict[str, int] = field(default_factory=dict)
    edges_by_relation: dict[str, int] = field(default_factory=dict)
    edges_with_spans: int = 0
    avg_confidence: float = 0.0
    loops_count: int = 0


class WorldModelCache:
    """In-memory cache for World Model data.
    
    Provides caching for:
    - Loop detection results (expensive graph algorithm)
    - Mechanism graph statistics
    - Frequent queries
    
    Cache is invalidated on edge insert or explicit invalidation.
    """
    
    def __init__(
        self,
        loop_cache_ttl: float = 300.0,  # 5 minutes
        stats_cache_ttl: float = 600.0,  # 10 minutes
        query_cache_ttl: float = 60.0,   # 1 minute
    ):
        self.loop_cache_ttl = loop_cache_ttl
        self.stats_cache_ttl = stats_cache_ttl
        self.query_cache_ttl = query_cache_ttl
        
        # Cache storage
        self._loops_cache: CacheEntry | None = None
        self._stats_cache: CacheEntry | None = None
        self._query_cache: dict[str, CacheEntry] = {}
        
        # Version counter for invalidation
        self._version: int = 0
        self._lock = asyncio.Lock()
    
    def _get_cache_key(self, *args) -> str:
        """Generate a cache key from arguments."""
        return ":".join(str(a) for a in args)
    
    async def get_loops(self) -> list[Any] | None:
        """Get cached loops if available and not expired.
        
        Returns:
            Cached loops or None if not cached/expired
        """
        if self._loops_cache is None:
            return None
        if self._loops_cache.is_expired():
            self._loops_cache = None
            return None
        return self._loops_cache.value
    
    async def set_loops(self, loops: list[Any]) -> None:
        """Cache detected loops.
        
        Args:
            loops: List of DetectedLoop objects
        """
        async with self._lock:
            self._loops_cache = CacheEntry(
                value=loops,
                created_at=time.time(),
                ttl_seconds=self.loop_cache_ttl,
            )
    
    async def get_stats(self) -> MechanismGraphStats | None:
        """Get cached graph statistics if available.
        
        Returns:
            Cached stats or None if not cached/expired
        """
        if self._stats_cache is None:
            return None
        if self._stats_cache.is_expired():
            self._stats_cache = None
            return None
        return self._stats_cache.value
    
    async def set_stats(self, stats: MechanismGraphStats) -> None:
        """Cache graph statistics.
        
        Args:
            stats: Graph statistics to cache
        """
        async with self._lock:
            self._stats_cache = CacheEntry(
                value=stats,
                created_at=time.time(),
                ttl_seconds=self.stats_cache_ttl,
            )
    
    async def get_query(self, key: str) -> Any | None:
        """Get cached query result.
        
        Args:
            key: Cache key for the query
            
        Returns:
            Cached result or None
        """
        entry = self._query_cache.get(key)
        if entry is None:
            return None
        if entry.is_expired():
            del self._query_cache[key]
            return None
        return entry.value
    
    async def set_query(self, key: str, value: Any) -> None:
        """Cache a query result.
        
        Args:
            key: Cache key
            value: Value to cache
        """
        async with self._lock:
            self._query_cache[key] = CacheEntry(
                value=value,
                created_at=time.time(),
                ttl_seconds=self.query_cache_ttl,
            )
    
    async def invalidate_loops(self) -> None:
        """Invalidate loop cache (call after edge insert)."""
        async with self._lock:
            self._loops_cache = None
            self._version += 1
    
    async def invalidate_stats(self) -> None:
        """Invalidate stats cache."""
        async with self._lock:
            self._stats_cache = None
    
    async def invalidate_all(self) -> None:
        """Invalidate all caches."""
        async with self._lock:
            self._loops_cache = None
            self._stats_cache = None
            self._query_cache.clear()
            self._version += 1
    
    def get_version(self) -> int:
        """Get current cache version (for debugging)."""
        return self._version


# Global cache instance
_cache: WorldModelCache | None = None


def get_world_model_cache() -> WorldModelCache:
    """Get the global world model cache instance.
    
    Returns:
        The global cache instance
    """
    global _cache
    if _cache is None:
        _cache = WorldModelCache()
    return _cache


async def get_cached_loops(session: AsyncSession) -> list[Any]:
    """Get loops from cache or compute and cache.
    
    This is the main entry point for cached loop access.
    
    Args:
        session: Database session
        
    Returns:
        List of detected loops
    """
    cache = get_world_model_cache()
    
    # Try cache first
    cached = await cache.get_loops()
    if cached is not None:
        return cached
    
    # Load persisted loops (fast path).
    # Reason: runtime must not run full cycle detection on every request.
    # Loops are mined offline and stored in `feedback_loop`.
    from apps.api.core.world_model.loop_reasoner import load_persisted_loops
    loops = await load_persisted_loops(session, max_loops=20)
    
    # Cache result
    await cache.set_loops(loops)
    
    return loops


async def get_cached_stats(session: AsyncSession) -> MechanismGraphStats:
    """Get graph statistics from cache or compute and cache.
    
    Args:
        session: Database session
        
    Returns:
        Mechanism graph statistics
    """
    cache = get_world_model_cache()
    
    # Try cache first
    cached = await cache.get_stats()
    if cached is not None:
        return cached
    
    # Compute stats
    stats = await _compute_graph_stats(session)
    
    # Cache result
    await cache.set_stats(stats)
    
    return stats


async def _compute_graph_stats(session: AsyncSession) -> MechanismGraphStats:
    """Compute mechanism graph statistics from database.
    
    Args:
        session: Database session
        
    Returns:
        Computed statistics
    """
    stats = MechanismGraphStats()
    
    # Count nodes
    result = await session.execute(
        text("SELECT COUNT(*) AS cnt FROM mechanism_node")
    )
    row = result.fetchone()
    stats.total_nodes = int(row.cnt) if row else 0
    
    # Count edges
    result = await session.execute(
        text("SELECT COUNT(*) AS cnt FROM mechanism_edge")
    )
    row = result.fetchone()
    stats.total_edges = int(row.cnt) if row else 0
    
    # Edges by relation type
    result = await session.execute(
        text("""
            SELECT relation_type, COUNT(*) AS cnt
            FROM mechanism_edge
            GROUP BY relation_type
        """)
    )
    for row in result.fetchall():
        stats.edges_by_relation[str(row.relation_type)] = int(row.cnt)
    
    # Edges with spans
    result = await session.execute(
        text("""
            SELECT COUNT(DISTINCT edge_id) AS cnt
            FROM mechanism_edge_span
        """)
    )
    row = result.fetchone()
    stats.edges_with_spans = int(row.cnt) if row else 0
    
    # Average confidence
    result = await session.execute(
        text("SELECT AVG(confidence) AS avg_conf FROM mechanism_edge")
    )
    row = result.fetchone()
    if row and row.avg_conf is not None:
        stats.avg_confidence = round(float(row.avg_conf), 3)
    
    # Count loops
    result = await session.execute(
        text("SELECT COUNT(*) AS cnt FROM feedback_loop")
    )
    row = result.fetchone()
    stats.loops_count = int(row.cnt) if row else 0
    
    # Edges by pillar (from mechanism_node)
    result = await session.execute(
        text("""
            SELECT n.ref_id, COUNT(DISTINCT e.id) AS cnt
            FROM mechanism_edge e
            JOIN mechanism_node n ON (e.from_node = n.id OR e.to_node = n.id)
            WHERE n.ref_kind = 'pillar'
            GROUP BY n.ref_id
        """)
    )
    for row in result.fetchall():
        stats.edges_by_pillar[str(row.ref_id)] = int(row.cnt)
    
    return stats


async def invalidate_on_edge_insert() -> None:
    """Call this after inserting mechanism edges to invalidate caches."""
    cache = get_world_model_cache()
    await cache.invalidate_loops()
    await cache.invalidate_stats()
