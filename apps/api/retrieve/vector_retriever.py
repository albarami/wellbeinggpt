"""
Vector Retriever Module

Retrieves evidence packets using vector similarity search with pgvector.
"""

from typing import Any, Optional

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession


async def search_similar_chunks(
    session: AsyncSession,
    query_embedding: list[float],
    top_k: int = 10,
    entity_types: Optional[list[str]] = None,
    chunk_types: Optional[list[str]] = None,
    threshold: float = 0.7,
) -> list[dict[str, Any]]:
    """
    Search for similar chunks using vector similarity.

    Args:
        session: Database session.
        query_embedding: Query vector.
        top_k: Number of results to return.
        entity_types: Optional filter for entity types.
        chunk_types: Optional filter for chunk types.
        threshold: Minimum similarity threshold.

    Returns:
        List of evidence packets with similarity scores.
    """
    # Build the query with optional filters
    query = """
        SELECT 
            c.chunk_id,
            c.entity_type,
            c.entity_id,
            c.chunk_type,
            c.text_ar,
            c.source_doc_id,
            c.source_anchor,
            1 - (e.vector <=> :query_vector) as similarity
        FROM chunk c
        JOIN embedding e ON e.chunk_id = c.chunk_id
        WHERE 1 - (e.vector <=> :query_vector) >= :threshold
    """

    params: dict[str, Any] = {
        "query_vector": str(query_embedding),
        "threshold": threshold,
    }

    if entity_types:
        query += " AND c.entity_type = ANY(:entity_types)"
        params["entity_types"] = entity_types

    if chunk_types:
        query += " AND c.chunk_type = ANY(:chunk_types)"
        params["chunk_types"] = chunk_types

    query += """
        ORDER BY similarity DESC
        LIMIT :top_k
    """
    params["top_k"] = top_k

    result = await session.execute(text(query), params)
    rows = result.fetchall()

    return [
        {
            "chunk_id": row.chunk_id,
            "entity_type": row.entity_type,
            "entity_id": row.entity_id,
            "chunk_type": row.chunk_type,
            "text_ar": row.text_ar,
            "source_doc_id": row.source_doc_id,
            "source_anchor": row.source_anchor,
            "refs": [],
            "similarity": row.similarity,
        }
        for row in rows
    ]


async def get_embedding_for_chunk(
    session: AsyncSession,
    chunk_id: str,
) -> Optional[list[float]]:
    """
    Get the embedding vector for a chunk.

    Args:
        session: Database session.
        chunk_id: Chunk ID.

    Returns:
        Embedding vector or None.
    """
    result = await session.execute(
        text("SELECT vector FROM embedding WHERE chunk_id = :chunk_id"),
        {"chunk_id": chunk_id}
    )
    row = result.fetchone()

    if row and row.vector:
        return list(row.vector)

    return None


async def store_embedding(
    session: AsyncSession,
    chunk_id: str,
    vector: list[float],
    model: str,
    dims: int,
) -> str:
    """
    Store an embedding for a chunk.

    Args:
        session: Database session.
        chunk_id: Chunk ID.
        vector: Embedding vector.
        model: Model name used.
        dims: Vector dimensions.

    Returns:
        Embedding ID.
    """
    import uuid

    embedding_id = str(uuid.uuid4())

    await session.execute(
        text("""
            INSERT INTO embedding (id, chunk_id, vector, model, dims)
            VALUES (:id, :chunk_id, :vector, :model, :dims)
            ON CONFLICT (chunk_id) DO UPDATE SET
                vector = EXCLUDED.vector,
                model = EXCLUDED.model
        """),
        {
            "id": embedding_id,
            "chunk_id": chunk_id,
            "vector": str(vector),
            "model": model,
            "dims": dims,
        }
    )

    return embedding_id


class VectorRetriever:
    """
    High-level vector retrieval interface.

    Handles embedding generation and similarity search.
    """

    def __init__(self, embedding_model: str = "text-embedding-3-large"):
        """
        Initialize the vector retriever.

        Args:
            embedding_model: Name of embedding model to use.
        """
        self.embedding_model = embedding_model
        self._embedding_dims = 3072  # Default for text-embedding-3-large

    async def embed_query(self, query: str) -> list[float]:
        """
        Generate embedding for a query.

        This is a placeholder - actual implementation requires API call.

        Args:
            query: Query text.

        Returns:
            Query embedding vector.
        """
        # TODO: Implement actual embedding generation via Azure OpenAI
        # For now, return a placeholder
        return [0.0] * self._embedding_dims

    async def search(
        self,
        session: AsyncSession,
        query: str,
        top_k: int = 10,
        entity_types: Optional[list[str]] = None,
        chunk_types: Optional[list[str]] = None,
    ) -> list[dict[str, Any]]:
        """
        Search for relevant chunks.

        Args:
            session: Database session.
            query: Query text.
            top_k: Number of results.
            entity_types: Optional entity type filter.
            chunk_types: Optional chunk type filter.

        Returns:
            List of evidence packets.
        """
        query_embedding = await self.embed_query(query)

        return await search_similar_chunks(
            session,
            query_embedding,
            top_k=top_k,
            entity_types=entity_types,
            chunk_types=chunk_types,
        )

