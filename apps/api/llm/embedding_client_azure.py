"""
Azure Embeddings Client

Enterprise-grade notes:
- Uses Azure OpenAI embeddings endpoint (deployment name configured via env).
- Supports a deterministic mock mode for tests (no network).
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from typing import List


@dataclass
class EmbeddingConfig:
    provider: str  # azure | mock
    azure_endpoint: str
    azure_api_key: str
    azure_api_version: str
    embedding_deployment: str
    dims: int

    @classmethod
    def from_env(cls) -> "EmbeddingConfig":
        provider = os.getenv("EMBEDDING_PROVIDER", "azure").lower()
        dims = int(os.getenv("EMBEDDING_DIMENSIONS", "3072"))
        return cls(
            provider=provider,
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT", ""),
            azure_api_key=os.getenv("AZURE_OPENAI_API_KEY", ""),
            azure_api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-10-21"),
            # IMPORTANT: Azure embeddings require a *deployment name*, not a model family name.
            # Do not fall back to EMBEDDING_MODEL, because it causes confusing 404 DeploymentNotFound
            # when users haven't created an embeddings deployment.
            embedding_deployment=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME", ""),
            dims=dims,
        )

    def is_configured(self) -> bool:
        if self.provider == "mock":
            return True
        return bool(self.azure_endpoint and self.azure_api_key and self.embedding_deployment)


def _mock_vector(text: str, dims: int) -> List[float]:
    """
    Deterministic pseudo-vector for tests.
    """
    h = hashlib.sha256(text.encode("utf-8")).digest()
    # Repeat hash bytes to fill dims
    out: list[float] = []
    for i in range(dims):
        b = h[i % len(h)]
        out.append((b / 255.0) * 2.0 - 1.0)
    return out


class AzureEmbeddingClient:
    def __init__(self, config: EmbeddingConfig | None = None):
        self.config = config or EmbeddingConfig.from_env()
        self._client = None

    async def _get_client(self):
        if self._client is None:
            from openai import AsyncAzureOpenAI

            self._client = AsyncAzureOpenAI(
                azure_endpoint=self.config.azure_endpoint,
                api_key=self.config.azure_api_key,
                api_version=self.config.azure_api_version,
            )
        return self._client

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        if self.config.provider == "mock":
            return [_mock_vector(t, self.config.dims) for t in texts]

        if not self.config.is_configured():
            raise RuntimeError("Embedding client is not configured (missing Azure env vars).")

        client = await self._get_client()
        resp = await client.embeddings.create(
            model=self.config.embedding_deployment,
            input=texts,
        )
        # SDK returns list under resp.data
        vectors: list[list[float]] = []
        for item in resp.data:
            vectors.append(list(item.embedding))
        return vectors


