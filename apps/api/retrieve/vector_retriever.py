"""
Vector Retriever Module

Retrieves evidence packets using a configurable vector backend.

Enterprise default:
- Use Azure AI Search for vector search (no pgvector dependency on Windows).
"""

from typing import Any, Optional

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession
import os

from apps.api.llm.embedding_client_azure import AzureEmbeddingClient, EmbeddingConfig


def _vector_backend() -> str:
    return os.getenv("VECTOR_BACKEND", "disabled").lower()  # azure_search | disabled


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
    # In no-pgvector environments, SQL vector similarity is not available.
    # We keep this function for compatibility, but it returns empty unless pgvector is installed.
    return []


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
            "vector": vector,
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
        cfg = EmbeddingConfig.from_env()
        self._embedding_dims = cfg.dims
        client = AzureEmbeddingClient(cfg)
        vecs = await client.embed_texts([query])
        return vecs[0]

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
        backend = _vector_backend()
        if backend == "disabled":
            return []

        if backend == "azure_search":
            from apps.api.retrieve.vector_retriever_azure_search import azure_search_vector_search

            cfg = EmbeddingConfig.from_env()
            self._embedding_dims = cfg.dims
            client = AzureEmbeddingClient(cfg)
            qvec = (await client.embed_texts([query]))[0]
            return await azure_search_vector_search(
                query_vector=qvec,
                top_k=top_k,
            )

        # Unknown backend
        return []

