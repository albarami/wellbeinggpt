"""
Azure AI Search Vector Retriever (REST)

This module is only used when VECTOR_BACKEND=azure_search and the required
AZURE_SEARCH_* env vars are set.
"""

from __future__ import annotations

import json
import os
from typing import Any

import httpx


def _cfg() -> dict[str, str]:
    return {
        "endpoint": os.getenv("AZURE_SEARCH_ENDPOINT", "").rstrip("/"),
        "api_key": os.getenv("AZURE_SEARCH_API_KEY", ""),
        "index": os.getenv("AZURE_SEARCH_INDEX_NAME", "wellbeing-chunks"),
        "api_version": os.getenv("AZURE_SEARCH_API_VERSION", "2023-11-01"),
    }


async def azure_search_vector_search(
    query_vector: list[float],
    top_k: int = 10,
) -> list[dict[str, Any]]:
    cfg = _cfg()
    if not cfg["endpoint"] or not cfg["api_key"] or not cfg["index"]:
        return []

    url = f"{cfg['endpoint']}/indexes/{cfg['index']}/docs/search?api-version={cfg['api_version']}"
    headers = {"api-key": cfg["api_key"], "content-type": "application/json"}

    # Minimal vector-only search. Assumes index contains:
    # - chunk_id, entity_type, entity_id, chunk_type, text_ar, source_doc_id, source_anchor, refs_json, vector
    body = {
        "count": False,
        "top": top_k,
        "vectorQueries": [
            {
                "kind": "vector",
                "vector": query_vector,
                "fields": "vector",
                "k": top_k,
            }
        ],
        "select": "chunk_id,entity_type,entity_id,chunk_type,text_ar,source_doc_id,source_anchor,refs_json",
    }

    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.post(url, headers=headers, json=body)
        if r.status_code >= 400:
            return []
        data = r.json()

    out: list[dict[str, Any]] = []
    for doc in data.get("value", []) or []:
        refs: list[dict[str, Any]] = []
        try:
            if doc.get("refs_json"):
                refs = json.loads(doc.get("refs_json"))
        except Exception:
            refs = []
        out.append(
            {
                "chunk_id": doc.get("chunk_id", ""),
                "entity_type": doc.get("entity_type", ""),
                "entity_id": doc.get("entity_id", ""),
                "chunk_type": doc.get("chunk_type", ""),
                "text_ar": doc.get("text_ar", ""),
                "source_doc_id": doc.get("source_doc_id", ""),
                "source_anchor": doc.get("source_anchor", ""),
                "refs": refs,
                "similarity": doc.get("@search.score", 0.0),
            }
        )
    return out


