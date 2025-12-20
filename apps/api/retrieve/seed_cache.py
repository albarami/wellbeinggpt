"""
Seed Cache: Deterministic, cached seed retrieval for synthesis intents.

Purpose:
- Eliminate nondeterminism in seed retrieval (same seeds every time)
- Cache database results to prevent "warmup variance"
- Ensure global_synthesis, cross_pillar, etc. never abstain due to missing seeds

Cache key: question_hash (for query-specific caching) or None (for global seeds)
"""

import hashlib
import logging
from typing import Any, Optional
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class SeedBundle:
    """Cached bundle of seed evidence packets."""
    pillar_definitions: list[dict[str, Any]] = field(default_factory=list)
    cross_pillar_edges: list[dict[str, Any]] = field(default_factory=list)
    policy_packet: Optional[dict[str, Any]] = None
    
    @property
    def all_packets(self) -> list[dict[str, Any]]:
        """Get all packets in deterministic order."""
        packets = []
        packets.extend(sorted(self.pillar_definitions, key=lambda p: p.get("entity_id", "")))
        packets.extend(sorted(self.cross_pillar_edges, key=lambda p: p.get("edge_id", "")))
        if self.policy_packet:
            packets.append(self.policy_packet)
        return packets
    
    @property
    def is_empty(self) -> bool:
        return not self.pillar_definitions and not self.cross_pillar_edges


class SeedCache:
    """
    In-memory cache for seed retrieval bundles.
    
    Thread-safe singleton pattern for process-level caching.
    """
    
    _instance: Optional["SeedCache"] = None
    _global_bundle: Optional[SeedBundle] = None
    _query_bundles: dict[str, SeedBundle] = {}
    _initialized: bool = False
    
    def __new__(cls) -> "SeedCache":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._query_bundles = {}
            cls._instance._global_bundle = None
            cls._instance._initialized = False
        return cls._instance
    
    @classmethod
    def get_instance(cls) -> "SeedCache":
        """Get the singleton cache instance."""
        return cls()
    
    @staticmethod
    def _hash_question(question: str) -> str:
        """Create a deterministic hash for caching."""
        return hashlib.sha256(question.encode("utf-8")).hexdigest()[:16]
    
    def get_global_bundle(self) -> Optional[SeedBundle]:
        """Get the cached global seed bundle (pillar definitions, cross-pillar edges)."""
        return self._global_bundle
    
    def set_global_bundle(self, bundle: SeedBundle) -> None:
        """Cache the global seed bundle."""
        self._global_bundle = bundle
        self._initialized = True
        logger.debug(f"Cached global seed bundle: {len(bundle.pillar_definitions)} pillars, {len(bundle.cross_pillar_edges)} edges")
    
    def get_query_bundle(self, question: str) -> Optional[SeedBundle]:
        """Get cached bundle for a specific question."""
        key = self._hash_question(question)
        return self._query_bundles.get(key)
    
    def set_query_bundle(self, question: str, bundle: SeedBundle) -> None:
        """Cache bundle for a specific question."""
        key = self._hash_question(question)
        self._query_bundles[key] = bundle
        # Limit cache size to prevent memory bloat
        if len(self._query_bundles) > 1000:
            # Remove oldest entries (simple LRU approximation)
            keys_to_remove = list(self._query_bundles.keys())[:500]
            for k in keys_to_remove:
                del self._query_bundles[k]
    
    def is_initialized(self) -> bool:
        """Check if global bundle has been loaded."""
        return self._initialized and self._global_bundle is not None
    
    def clear(self) -> None:
        """Clear all caches (for testing)."""
        self._global_bundle = None
        self._query_bundles = {}
        self._initialized = False


async def load_global_seed_bundle(session) -> SeedBundle:
    """
    Load the global seed bundle from database.
    
    This should be called once at startup or on first request.
    Results are cached for subsequent requests.
    
    Returns:
        SeedBundle with pillar definitions and cross-pillar edges
    """
    from sqlalchemy import text
    
    bundle = SeedBundle()
    
    # 1. Fetch pillar definition chunks (one per pillar, deterministic order)
    try:
        pillar_rows = (await session.execute(text("""
            SELECT DISTINCT ON (c.entity_id)
                c.chunk_id as chunk_id,
                c.text_ar,
                c.chunk_type,
                p.id as pillar_id,
                p.name_ar as pillar_name_ar
            FROM chunk c
            JOIN pillar p ON c.entity_id = p.id
            WHERE c.entity_type = 'pillar' AND c.chunk_type = 'definition'
            ORDER BY c.entity_id, c.chunk_id
            LIMIT 5
        """))).fetchall()
        
        for row in pillar_rows:
            bundle.pillar_definitions.append({
                "chunk_id": str(row.chunk_id),
                "text_ar": row.text_ar,
                "chunk_type": row.chunk_type,
                "entity_type": "pillar",
                "entity_id": str(row.pillar_id),
                "entity_name_ar": row.pillar_name_ar,
                "source": "seed_floor",
            })
        
        logger.debug(f"Loaded {len(bundle.pillar_definitions)} pillar definitions")
        
    except Exception as e:
        logger.warning(f"Failed to load pillar definitions: {e}")
    
    # 2. Fetch cross-pillar edges (deterministic order by edge_id)
    try:
        edge_rows = (await session.execute(text("""
            SELECT 
                e.edge_id,
                e.relation_type,
                e.from_entity_id,
                e.to_entity_id,
                ej.quote as justification_quote,
                c.chunk_id as source_chunk_id,
                c.text_ar as chunk_text_ar
            FROM edge e
            LEFT JOIN edge_justification_span ej ON ej.edge_id = e.edge_id
            LEFT JOIN chunk c ON c.chunk_id = ej.chunk_id
            WHERE e.from_entity_type = 'pillar' AND e.to_entity_type = 'pillar'
            ORDER BY e.edge_id ASC
            LIMIT 10
        """))).fetchall()
        
        for row in edge_rows:
            text_ar = row.justification_quote or row.chunk_text_ar or f"({row.relation_type}) {row.from_entity_id} → {row.to_entity_id}"
            bundle.cross_pillar_edges.append({
                "chunk_id": f"edge_{row.edge_id}",
                "text_ar": text_ar,
                "chunk_type": "cross_pillar_edge",
                "source": "seed_floor",
                "edge_id": str(row.edge_id),
            })
        
        logger.debug(f"Loaded {len(bundle.cross_pillar_edges)} cross-pillar edges")
        
    except Exception as e:
        logger.debug(f"Failed to load cross-pillar edges (may not exist): {e}")
    
    return bundle


def get_policy_packet() -> dict[str, Any]:
    """
    Get the deterministic policy response packet for system_limits_policy intent.
    """
    return {
        "chunk_id": "system_policy_response",
        "text_ar": (
            "حدود الربط في النظام:\n"
            "1. لا يُنشئ النظام روابط سببية بين الركائز إلا إذا وُجد نص صريح يدعمها.\n"
            "2. عند عدم وجود نص، يُصرّح بذلك: \"غير منصوص عليه في الإطار\".\n"
            "3. يعتمد النظام على الأدلة النصية المؤصّلة فقط.\n"
            "4. لا يُقدّم النظام استشارات طبية أو نفسية تشخيصية."
        ),
        "chunk_type": "policy",
        "source": "system_policy",
    }
