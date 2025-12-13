"""
Azure env fallback helpers.

Some deployments provide Azure OpenAI configuration using non-standard .env keys like:
- ENDPOINT_5 (full chat/completions URL including deployment + api-version)
- API_KEY_5
- API_VERSION_5
- DEPLOYMENT_5

This module normalizes those into the canonical env shape used by the codebase:
- AZURE_OPENAI_ENDPOINT (base resource endpoint)
- AZURE_OPENAI_API_KEY
- AZURE_OPENAI_API_VERSION
- AZURE_OPENAI_DEPLOYMENT_NAME
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional
from urllib.parse import urlparse, parse_qs


@dataclass(frozen=True)
class AzureNormalizedConfig:
    endpoint: str
    api_key: str
    api_version: str
    deployment_name: str


def _strip_quotes(v: str) -> str:
    v = (v or "").strip()
    if (len(v) >= 2) and ((v[0] == v[-1]) and v[0] in ("'", '"')):
        return v[1:-1].strip()
    return v


def _normalize_base_endpoint(url: str) -> str:
    url = _strip_quotes(url)
    if not url:
        return ""
    # If provided a full Azure chat/completions URL, reduce to scheme://host/
    p = urlparse(url)
    if p.scheme and p.netloc:
        return f"{p.scheme}://{p.netloc}/"
    return url if url.endswith("/") else (url + "/")


def _parse_deployment_from_path(url: str) -> str:
    url = _strip_quotes(url)
    if not url:
        return ""
    p = urlparse(url)
    parts = [x for x in (p.path or "").split("/") if x]
    # Expected: /openai/deployments/<deployment>/chat/completions
    if "deployments" in parts:
        i = parts.index("deployments")
        if i + 1 < len(parts):
            return parts[i + 1]
    return ""


def _parse_api_version_from_url(url: str) -> str:
    url = _strip_quotes(url)
    if not url:
        return ""
    p = urlparse(url)
    q = parse_qs(p.query or "")
    v = (q.get("api-version") or [""])[0]
    return _strip_quotes(v)


def load_azure_normalized_config_from_env() -> Optional[AzureNormalizedConfig]:
    """
    Load Azure config from either canonical AZURE_OPENAI_* vars or fallback *_5 vars.
    """
    # Canonical
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "") or ""
    api_key = os.getenv("AZURE_OPENAI_API_KEY", "") or ""
    api_version = os.getenv("AZURE_OPENAI_API_VERSION", "") or ""
    deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "") or ""
    if endpoint and api_key and deployment:
        return AzureNormalizedConfig(
            endpoint=_normalize_base_endpoint(endpoint),
            api_key=api_key.strip(),
            api_version=(api_version.strip() or "2024-10-21"),
            deployment_name=deployment.strip(),
        )

    # Fallback (user-provided .env style)
    endpoint_full = os.getenv("ENDPOINT_5", "") or os.getenv("ENDPOINT_5_0", "") or ""
    api_key_5 = os.getenv("API_KEY_5", "") or ""
    deployment_5 = _strip_quotes(os.getenv("DEPLOYMENT_5", "") or "")
    api_version_5 = _strip_quotes(os.getenv("API_VERSION_5", "") or "")

    # Derive from full endpoint URL if present
    derived_deployment = _parse_deployment_from_path(endpoint_full)
    derived_api_version = _parse_api_version_from_url(endpoint_full)
    base_endpoint = _normalize_base_endpoint(endpoint_full)

    deployment_name = deployment_5 or derived_deployment
    api_version = derived_api_version or api_version_5 or "2024-10-21"

    if base_endpoint and api_key_5 and deployment_name:
        return AzureNormalizedConfig(
            endpoint=base_endpoint,
            api_key=api_key_5.strip(),
            api_version=api_version,
            deployment_name=deployment_name,
        )

    return None


