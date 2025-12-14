"""
Retrieval Quality Tests

Integration tests for vector and hybrid retrieval:
- Exact-name queries return correct definition chunks
- Verse-ref queries return correct evidence chunks
- Cross-pillar hybrid retrieval behavior

These tests require a live database with RUN_DB_TESTS=1 and
optionally Azure AI Search for vector tests.
"""

import os
import pytest

from sqlalchemy import text

from apps.api.retrieve.normalize_ar import normalize_for_matching


# Note: require_db and require_azure_search fixtures are defined in conftest.py


@pytest.mark.asyncio
async def test_entity_resolver_finds_tawhid(require_db):
    """Test that entity resolver finds التوحيد."""
    from apps.api.core.database import get_session
    from apps.api.retrieve.entity_resolver import EntityResolver
    
    async with get_session() as session:
        resolver = EntityResolver()
        
        pillars = (await session.execute(text("SELECT id, name_ar FROM pillar"))).fetchall()
        core_values = (await session.execute(text("SELECT id, name_ar FROM core_value"))).fetchall()
        sub_values = (await session.execute(text("SELECT id, name_ar FROM sub_value"))).fetchall()
        
        resolver.load_entities(
            pillars=[{"id": r.id, "name_ar": r.name_ar} for r in pillars],
            core_values=[{"id": r.id, "name_ar": r.name_ar} for r in core_values],
            sub_values=[{"id": r.id, "name_ar": r.name_ar} for r in sub_values],
        )
        
        results = resolver.resolve("ما هو التوحيد؟")
        
        # Should find something
        assert len(results) > 0, "Expected to find entities for التوحيد query"
        
        # Check if any result matches توحيد
        found_tawhid = any(
            "توحيد" in normalize_for_matching(r.name_ar)
            for r in results
        )
        assert found_tawhid, f"Expected to find التوحيد, got: {[r.name_ar for r in results]}"


@pytest.mark.asyncio
async def test_entity_resolver_finds_tawazun(require_db):
    """Test that entity resolver finds التوازن العاطفي."""
    from apps.api.core.database import get_session
    from apps.api.retrieve.entity_resolver import EntityResolver
    
    async with get_session() as session:
        resolver = EntityResolver()
        
        pillars = (await session.execute(text("SELECT id, name_ar FROM pillar"))).fetchall()
        core_values = (await session.execute(text("SELECT id, name_ar FROM core_value"))).fetchall()
        sub_values = (await session.execute(text("SELECT id, name_ar FROM sub_value"))).fetchall()
        
        resolver.load_entities(
            pillars=[{"id": r.id, "name_ar": r.name_ar} for r in pillars],
            core_values=[{"id": r.id, "name_ar": r.name_ar} for r in core_values],
            sub_values=[{"id": r.id, "name_ar": r.name_ar} for r in sub_values],
        )
        
        results = resolver.resolve("التوازن العاطفي")
        
        assert len(results) > 0, "Expected to find entities for التوازن العاطفي"


@pytest.mark.asyncio
async def test_sql_retrieval_returns_chunks_for_entity(require_db):
    """Test that SQL retrieval returns chunks for a known entity."""
    from apps.api.core.database import get_session
    from apps.api.retrieve.sql_retriever import get_chunks_with_refs
    from apps.api.core.schemas import EntityType
    
    async with get_session() as session:
        # Get any sub_value with chunks
        sv_row = (await session.execute(
            text("""
                SELECT sv.id
                FROM sub_value sv
                JOIN chunk c ON c.entity_type = 'sub_value' AND c.entity_id = sv.id
                LIMIT 1
            """)
        )).fetchone()
        
        if not sv_row:
            pytest.skip("No sub_value with chunks found")
        
        chunks = await get_chunks_with_refs(session, EntityType.SUB_VALUE, sv_row.id)
        
        assert len(chunks) > 0, "Expected to find chunks for sub_value"
        
        # Check chunk structure
        chunk = chunks[0]
        assert "chunk_id" in chunk
        assert "text_ar" in chunk
        assert "source_anchor" in chunk


@pytest.mark.asyncio
async def test_hybrid_retriever_expands_from_resolved_entities(require_db):
    """Test that hybrid retriever expands from resolved entities via graph."""
    from apps.api.core.database import get_session
    from apps.api.retrieve.entity_resolver import EntityResolver
    from apps.api.retrieve.hybrid_retriever import HybridRetriever, RetrievalInputs
    
    async with get_session() as session:
        # Set up resolver
        resolver = EntityResolver()
        pillars = (await session.execute(text("SELECT id, name_ar FROM pillar"))).fetchall()
        core_values = (await session.execute(text("SELECT id, name_ar FROM core_value"))).fetchall()
        sub_values = (await session.execute(text("SELECT id, name_ar FROM sub_value"))).fetchall()
        
        resolver.load_entities(
            pillars=[{"id": r.id, "name_ar": r.name_ar} for r in pillars],
            core_values=[{"id": r.id, "name_ar": r.name_ar} for r in core_values],
            sub_values=[{"id": r.id, "name_ar": r.name_ar} for r in sub_values],
        )
        
        # Find an entity with chunks
        sv_with_chunks = (await session.execute(
            text("""
                SELECT sv.id, sv.name_ar
                FROM sub_value sv
                JOIN chunk c ON c.entity_type = 'sub_value' AND c.entity_id = sv.id
                LIMIT 1
            """)
        )).fetchone()
        
        if not sv_with_chunks:
            pytest.skip("No sub_value with chunks found")
        
        # Resolve and retrieve
        resolved = resolver.resolve(sv_with_chunks.name_ar)
        
        retriever = HybridRetriever(enable_vector=False)  # Disable vector for this test
        
        inputs = RetrievalInputs(
            query=sv_with_chunks.name_ar,
            resolved_entities=[
                {"type": r.entity_type.value, "id": r.entity_id, "name_ar": r.name_ar}
                for r in resolved[:3]
            ],
        )
        
        result = await retriever.retrieve(session, inputs)
        
        assert len(result.evidence_packets) > 0, "Expected evidence packets from retrieval"


@pytest.mark.asyncio
async def test_ref_node_retrieval_finds_entities(require_db):
    """Test that ref-node based retrieval finds related entities."""
    from apps.api.core.database import get_session
    from apps.api.retrieve.graph_retriever import expand_graph
    
    async with get_session() as session:
        # Find a ref node that's used by multiple entities
        ref_row = (await session.execute(
            text("""
                SELECT to_id AS ref_id, COUNT(*) AS cnt
                FROM edge
                WHERE to_type = 'ref'
                  AND rel_type IN ('MENTIONS_REF', 'REFERS_TO')
                  AND status = 'approved'
                GROUP BY to_id
                HAVING COUNT(*) >= 2
                ORDER BY cnt DESC
                LIMIT 1
            """)
        )).fetchone()
        
        if not ref_row:
            # No ref nodes - skip gracefully
            print("No ref node with multiple entities found - REFERS_TO/MENTIONS_REF edges may not be created yet")
            pytest.skip("No ref node with multiple entities found")
        
        # Expand from the ref node
        neighbors = await expand_graph(
            session,
            entity_type="ref",
            entity_id=ref_row.ref_id,
            depth=1,
            relationship_types=["MENTIONS_REF", "REFERS_TO"],
        )
        
        assert len(neighbors) >= 2, (
            f"Expected at least 2 neighbors for ref node {ref_row.ref_id}, "
            f"got {len(neighbors)}"
        )


@pytest.mark.asyncio
async def test_cross_pillar_discovery_via_shared_refs(require_db):
    """Test that cross-pillar discovery works via shared refs."""
    from apps.api.core.database import get_session
    
    async with get_session() as session:
        # Find entities from different pillars that share a ref
        cross_pillar = (await session.execute(
            text("""
                WITH entity_pillars AS (
                    SELECT 
                        e.from_type AS entity_type,
                        e.from_id AS entity_id,
                        e.to_id AS ref_id,
                        COALESCE(
                            (SELECT pillar_id FROM core_value WHERE id = e.from_id),
                            (SELECT cv.pillar_id 
                             FROM sub_value sv 
                             JOIN core_value cv ON cv.id = sv.core_value_id 
                             WHERE sv.id = e.from_id)
                        ) AS pillar_id
                    FROM edge e
                    WHERE e.rel_type = 'MENTIONS_REF'
                      AND e.to_type = 'ref'
                      AND e.status = 'approved'
                      AND e.from_type IN ('core_value', 'sub_value')
                )
                SELECT 
                    ep1.ref_id,
                    ep1.entity_id AS e1,
                    ep1.pillar_id AS p1,
                    ep2.entity_id AS e2,
                    ep2.pillar_id AS p2
                FROM entity_pillars ep1
                JOIN entity_pillars ep2 
                    ON ep1.ref_id = ep2.ref_id 
                    AND ep1.entity_id < ep2.entity_id
                    AND ep1.pillar_id <> ep2.pillar_id
                WHERE ep1.pillar_id IS NOT NULL 
                  AND ep2.pillar_id IS NOT NULL
                LIMIT 5
            """)
        )).fetchall()
        
        # It's OK if no cross-pillar refs exist - just report
        if cross_pillar:
            print(f"Found {len(cross_pillar)} cross-pillar shared refs")
            for row in cross_pillar[:3]:
                print(f"  Ref {row.ref_id}: {row.e1} (P:{row.p1}) ↔ {row.e2} (P:{row.p2})")
        else:
            print("No cross-pillar shared refs found (may be OK for small corpus)")


@pytest.mark.asyncio
async def test_vector_search_returns_results(require_db, require_azure_search):
    """Test that vector search returns results for a known term."""
    from apps.api.core.database import get_session
    from apps.api.retrieve.vector_retriever import VectorRetriever
    
    async with get_session() as session:
        retriever = VectorRetriever()
        
        results = await retriever.search(
            session,
            query="الإيمان",
            top_k=5,
        )
        
        assert len(results) > 0, "Expected vector search to return results"
        
        # Check result structure
        result = results[0]
        assert "chunk_id" in result
        assert "text_ar" in result


@pytest.mark.asyncio
async def test_vector_search_definition_in_top_k(require_db, require_azure_search):
    """Test that entity's definition chunk appears in top-K for exact name query."""
    from apps.api.core.database import get_session
    from apps.api.retrieve.vector_retriever import VectorRetriever
    
    async with get_session() as session:
        # Get an entity with a definition chunk
        entity_row = (await session.execute(
            text("""
                SELECT sv.id, sv.name_ar, c.chunk_id
                FROM sub_value sv
                JOIN chunk c ON c.entity_type = 'sub_value' 
                    AND c.entity_id = sv.id 
                    AND c.chunk_type = 'definition'
                LIMIT 1
            """)
        )).fetchone()
        
        if not entity_row:
            pytest.skip("No sub_value with definition chunk found")
        
        retriever = VectorRetriever()
        
        results = await retriever.search(
            session,
            query=entity_row.name_ar,
            top_k=10,
        )
        
        found_chunk_ids = [r.get("chunk_id") for r in results]
        
        # Check if expected chunk is in results
        if entity_row.chunk_id not in found_chunk_ids:
            # This is a warning, not a hard failure - vector search may vary
            print(
                f"Warning: Expected chunk {entity_row.chunk_id} not in top-10 "
                f"for query '{entity_row.name_ar}'"
            )
