"""
Entity Resolver Module

Resolves entity names from queries to their IDs using exact matching
and fuzzy matching with Arabic normalization.
"""

from dataclasses import dataclass
from typing import Optional

from apps.api.core.schemas import EntityType
from apps.api.retrieve.normalize_ar import normalize_for_matching


@dataclass
class ResolvedEntity:
    """A resolved entity from the query."""

    entity_type: EntityType
    entity_id: str
    name_ar: str
    match_type: str  # exact | normalized | fuzzy
    confidence: float  # 0.0 to 1.0


class EntityResolver:
    """
    Resolves entity names from queries.

    Uses a hierarchical matching strategy:
    1. Exact match
    2. Normalized match (diacritics removed, Alef normalized)
    3. Fuzzy match (prefix/suffix)
    """

    def __init__(self):
        """Initialize the entity resolver."""
        # In-memory entity index (loaded from DB)
        self._pillars: dict[str, str] = {}  # normalized_name -> id
        self._core_values: dict[str, str] = {}
        self._sub_values: dict[str, str] = {}

        # Original names for display
        self._names: dict[str, str] = {}  # id -> original_name_ar

    def load_entities(
        self,
        pillars: list[dict],
        core_values: list[dict],
        sub_values: list[dict],
    ) -> None:
        """
        Load entities into the resolver.

        Args:
            pillars: List of pillar dicts with id and name_ar.
            core_values: List of core value dicts.
            sub_values: List of sub-value dicts.
        """
        self._pillars = {}
        self._core_values = {}
        self._sub_values = {}
        self._names = {}

        for p in pillars:
            normalized = normalize_for_matching(p["name_ar"])
            self._pillars[normalized] = p["id"]
            self._names[p["id"]] = p["name_ar"]

        for cv in core_values:
            normalized = normalize_for_matching(cv["name_ar"])
            self._core_values[normalized] = cv["id"]
            self._names[cv["id"]] = cv["name_ar"]

        for sv in sub_values:
            normalized = normalize_for_matching(sv["name_ar"])
            self._sub_values[normalized] = sv["id"]
            self._names[sv["id"]] = sv["name_ar"]

    def resolve(self, query: str) -> list[ResolvedEntity]:
        """
        Resolve entities mentioned in a query.

        Args:
            query: The user's query text.

        Returns:
            List of resolved entities, sorted by confidence.
        """
        resolved = []
        normalized_query = normalize_for_matching(query)

        # Check for entity matches
        # Priority: pillars > core values > sub-values

        for name, entity_id in self._pillars.items():
            match_result = self._match(name, normalized_query)
            if match_result:
                resolved.append(ResolvedEntity(
                    entity_type=EntityType.PILLAR,
                    entity_id=entity_id,
                    name_ar=self._names[entity_id],
                    match_type=match_result[0],
                    confidence=match_result[1],
                ))

        for name, entity_id in self._core_values.items():
            match_result = self._match(name, normalized_query)
            if match_result:
                resolved.append(ResolvedEntity(
                    entity_type=EntityType.CORE_VALUE,
                    entity_id=entity_id,
                    name_ar=self._names[entity_id],
                    match_type=match_result[0],
                    confidence=match_result[1],
                ))

        for name, entity_id in self._sub_values.items():
            match_result = self._match(name, normalized_query)
            if match_result:
                resolved.append(ResolvedEntity(
                    entity_type=EntityType.SUB_VALUE,
                    entity_id=entity_id,
                    name_ar=self._names[entity_id],
                    match_type=match_result[0],
                    confidence=match_result[1],
                ))

        # Sort by confidence descending
        resolved.sort(key=lambda r: r.confidence, reverse=True)

        return resolved

    def _match(
        self, entity_name: str, normalized_query: str
    ) -> Optional[tuple[str, float]]:
        """
        Check if entity name matches in the query.

        Returns:
            Tuple of (match_type, confidence) or None.
        """
        # Exact match
        if entity_name in normalized_query:
            return ("exact", 1.0)

        # Word boundary match
        query_words = set(normalized_query.split())
        entity_words = set(entity_name.split())

        if entity_words and entity_words.issubset(query_words):
            return ("normalized", 0.9)

        # Partial match (at least one significant word)
        common = entity_words.intersection(query_words)
        if common and len(next(iter(common))) > 2:
            overlap = len(common) / len(entity_words)
            if overlap >= 0.5:
                return ("fuzzy", overlap * 0.7)

        return None

    def get_entity_name(self, entity_id: str) -> Optional[str]:
        """Get the original Arabic name for an entity ID."""
        return self._names.get(entity_id)


# Singleton resolver instance
_resolver: Optional[EntityResolver] = None


def get_resolver() -> EntityResolver:
    """Get the global entity resolver instance."""
    global _resolver
    if _resolver is None:
        _resolver = EntityResolver()
    return _resolver

