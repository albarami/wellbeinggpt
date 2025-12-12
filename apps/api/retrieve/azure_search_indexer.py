"""
Azure AI Search Indexer (REST)

Creates/updates an index for wellbeing chunks and upserts chunk documents.
Used during ingestion when VECTOR_BACKEND=azure_search.
"""

from __future__ import annotations

import json
import os
from typing import Any, Iterable

import httpx


def _cfg() -> dict[str, str]:
    return {
        "endpoint": os.getenv("AZURE_SEARCH_ENDPOINT", "").rstrip("/"),
        "api_key": os.getenv("AZURE_SEARCH_API_KEY", ""),
        "index": os.getenv("AZURE_SEARCH_INDEX_NAME", "wellbeing-chunks"),
        "api_version": os.getenv("AZURE_SEARCH_API_VERSION", "2023-11-01"),
    }


def is_configured() -> bool:
    cfg = _cfg()
    return bool(cfg["endpoint"] and cfg["api_key"] and cfg["index"])


async def ensure_index(dims: int) -> bool:
    """
    Ensure the Azure AI Search index exists.
    Returns True if index exists/created, False if not configured or failed.
    """
    cfg = _cfg()
    if not is_configured():
        return False

    base = cfg["endpoint"]
    index = cfg["index"]
    api_version = cfg["api_version"]
    headers = {"api-key": cfg["api_key"], "content-type": "application/json"}

    url_get = f"{base}/indexes/{index}?api-version={api_version}"
    url_put = f"{base}/indexes/{index}?api-version={api_version}"

    schema = {
        "name": index,
        "fields": [
            {"name": "chunk_id", "type": "Edm.String", "key": True, "filterable": True},
            {"name": "entity_type", "type": "Edm.String", "filterable": True},
            {"name": "entity_id", "type": "Edm.String", "filterable": True},
            {"name": "chunk_type", "type": "Edm.String", "filterable": True},
            {"name": "text_ar", "type": "Edm.String", "searchable": True},
            {"name": "source_doc_id", "type": "Edm.String", "filterable": True},
            {"name": "source_anchor", "type": "Edm.String", "filterable": True},
            # Keep refs as JSON string for simple schema portability.
            {"name": "refs_json", "type": "Edm.String", "searchable": False},
            # Vector field
            {
                "name": "vector",
                "type": "Collection(Edm.Single)",
                "searchable": True,
                "dimensions": dims,
                "vectorSearchProfile": "vs-profile",
            },
        ],
        "vectorSearch": {
            "algorithms": [
                {"name": "hnsw", "kind": "hnsw"},
            ],
            "profiles": [
                {"name": "vs-profile", "algorithm": "hnsw"},
            ],
        },
    }

    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.get(url_get, headers=headers)
        if r.status_code == 200:
            return True
        # Create/update
        r2 = await client.put(url_put, headers=headers, json=schema)
        return r2.status_code < 400


async def upsert_documents(docs: list[dict[str, Any]]) -> bool:
    """
    Upsert docs into the index using indexing actions.
    """
    cfg = _cfg()
    if not is_configured():
        return False

    base = cfg["endpoint"]
    index = cfg["index"]
    api_version = cfg["api_version"]
    headers = {"api-key": cfg["api_key"], "content-type": "application/json"}

    url = f"{base}/indexes/{index}/docs/index?api-version={api_version}"
    body = {
        "value": [
            {"@search.action": "mergeOrUpload", **d}
            for d in docs
        ]
    }

    async with httpx.AsyncClient(timeout=60) as client:
        r = await client.post(url, headers=headers, json=body)
        return r.status_code < 400


def chunk_doc(
    *,
    chunk_id: str,
    entity_type: str,
    entity_id: str,
    chunk_type: str,
    text_ar: str,
    source_doc_id: str,
    source_anchor: str,
    refs: list[dict[str, Any]],
    vector: list[float],
) -> dict[str, Any]:
    return {
        "chunk_id": chunk_id,
        "entity_type": entity_type,
        "entity_id": entity_id,
        "chunk_type": chunk_type,
        "text_ar": text_ar,
        "source_doc_id": source_doc_id,
        "source_anchor": source_anchor,
        "refs_json": json.dumps(refs, ensure_ascii=False),
        "vector": vector,
    }


