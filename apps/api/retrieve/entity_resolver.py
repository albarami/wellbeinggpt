"""
Entity Resolver Module

Resolves entity names from queries to their IDs using exact matching
and fuzzy matching with Arabic normalization.
"""

from dataclasses import dataclass
from typing import Optional
import json
from pathlib import Path

from apps.api.core.schemas import EntityType
from apps.api.retrieve.normalize_ar import normalize_for_matching
from apps.api.retrieve.arabic_morph import expand_query_terms, phrase_variants


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

        # Optional alias index (normalized alias -> id)
        self._aliases: dict[str, str] = {}

    def load_entities(
        self,
        pillars: list[dict],
        core_values: list[dict],
        sub_values: list[dict],
        aliases_path: Optional[str | Path] = None,
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
        self._aliases = {}

        for p in pillars:
            normalized = normalize_for_matching(p["name_ar"])
            self._pillars[normalized] = p["id"]
            for v in phrase_variants(p["name_ar"]):
                self._pillars.setdefault(v, p["id"])
            self._names[p["id"]] = p["name_ar"]

        for cv in core_values:
            normalized = normalize_for_matching(cv["name_ar"])
            self._core_values[normalized] = cv["id"]
            for v in phrase_variants(cv["name_ar"]):
                self._core_values.setdefault(v, cv["id"])
            self._names[cv["id"]] = cv["name_ar"]

        for sv in sub_values:
            normalized = normalize_for_matching(sv["name_ar"])
            self._sub_values[normalized] = sv["id"]
            for v in phrase_variants(sv["name_ar"]):
                self._sub_values.setdefault(v, sv["id"])
            self._names[sv["id"]] = sv["name_ar"]

        # Optional aliases file (Arabic-first)
        if aliases_path:
            self._load_aliases(aliases_path)

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
        expanded_terms = expand_query_terms(query)

        # Check for entity matches
        # Priority: pillars > core values > sub-values

        for name, entity_id in self._pillars.items():
            match_result = self._match(name, normalized_query)
            if not match_result and expanded_terms:
                match_result = self._match_terms(name, expanded_terms)
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
            if not match_result and expanded_terms:
                match_result = self._match_terms(name, expanded_terms)
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
            if not match_result and expanded_terms:
                match_result = self._match_terms(name, expanded_terms)
            if match_result:
                resolved.append(ResolvedEntity(
                    entity_type=EntityType.SUB_VALUE,
                    entity_id=entity_id,
                    name_ar=self._names[entity_id],
                    match_type=match_result[0],
                    confidence=match_result[1],
                ))

        # Alias matches (all entity types, but low priority)
        for alias, entity_id in self._aliases.items():
            match_result = self._match(alias, normalized_query) or self._match_terms(alias, expanded_terms)
            if match_result:
                et = self._infer_entity_type(entity_id)
                if not et:
                    continue
                resolved.append(
                    ResolvedEntity(
                        entity_type=et,
                        entity_id=entity_id,
                        name_ar=self._names.get(entity_id, alias),
                        match_type="alias_" + match_result[0],
                        confidence=min(match_result[1], 0.85),
                    )
                )

        # Deduplicate by (type,id) while keeping the best confidence.
        best: dict[tuple[str, str], ResolvedEntity] = {}
        for r in resolved:
            key = (r.entity_type.value, r.entity_id)
            if key not in best or r.confidence > best[key].confidence:
                best[key] = r

        out = list(best.values())
        out.sort(key=lambda r: r.confidence, reverse=True)
        return out

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

    def _match_terms(self, entity_name: str, expanded_terms: set[str]) -> Optional[tuple[str, float]]:
        """
        Match entity name using expanded token variants.

        Returns:
            (match_type, confidence) or None.
        """
        if not expanded_terms:
            return None

        entity_terms = set(entity_name.split())
        if not entity_terms:
            return None

        common = entity_terms.intersection(expanded_terms)
        if not common:
            return None

        overlap = len(common) / max(len(entity_terms), 1)
        if overlap >= 0.6:
            return ("morph", 0.88)
        if overlap >= 0.4:
            return ("morph", 0.75)
        return None

    def _infer_entity_type(self, entity_id: str) -> Optional[EntityType]:
        """
        Best-effort inference for alias targets based on loaded name map.
        """
        if entity_id in self._pillars.values():
            return EntityType.PILLAR
        if entity_id in self._core_values.values():
            return EntityType.CORE_VALUE
        if entity_id in self._sub_values.values():
            return EntityType.SUB_VALUE
        return None

    def _load_aliases(self, aliases_path: str | Path) -> None:
        """
        Load alias mappings from a JSON file.

        File format:
          [{"alias_ar": "...", "entity_id": "..."}]
        """
        path = Path(aliases_path)
        if not path.exists():
            return
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return
        if not isinstance(data, list):
            return
        for item in data:
            if not isinstance(item, dict):
                continue
            alias = item.get("alias_ar")
            entity_id = item.get("entity_id")
            if not alias or not entity_id:
                continue
            self._aliases[normalize_for_matching(alias)] = str(entity_id)

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

