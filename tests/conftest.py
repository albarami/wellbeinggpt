"""
Pytest configuration and fixtures.

This file is automatically loaded by pytest before running tests.
It loads environment variables from .env file and provides common fixtures.
"""

import os
from pathlib import Path

import pytest


def pytest_configure(config):
    """Load environment variables from .env file before tests run."""
    env_file = Path(__file__).parent.parent / ".env"
    
    if env_file.exists():
        print(f"\n[conftest] Loading environment from: {env_file}")
        with open(env_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                # Skip comments and empty lines
                if not line or line.startswith("#"):
                    continue
                # Parse KEY=VALUE
                if "=" in line:
                    key, _, value = line.partition("=")
                    key = key.strip()
                    value = value.strip()
                    # Don't override existing env vars
                    if key and key not in os.environ:
                        os.environ[key] = value
        
        # Always enable DB tests when .env is loaded
        os.environ.setdefault("RUN_DB_TESTS", "1")
        
        print(f"[conftest] DATABASE_URL configured: {bool(os.getenv('DATABASE_URL'))}")
        print(f"[conftest] AZURE_OPENAI_ENDPOINT configured: {bool(os.getenv('AZURE_OPENAI_ENDPOINT'))}")
        print(f"[conftest] AZURE_SEARCH_ENDPOINT configured: {bool(os.getenv('AZURE_SEARCH_ENDPOINT'))}")
        print(f"[conftest] VECTOR_BACKEND: {os.getenv('VECTOR_BACKEND', 'disabled')}")
    else:
        print(f"\n[conftest] No .env file found at: {env_file}")

    # Best-effort: ensure the framework is ingested into the DB for integration tests.
    # Reason: integration tests assume a populated DB; this keeps the DB consistent
    # with the current code (e.g., newly added chunks).
    if os.getenv("RUN_DB_TESTS") == "1" and os.getenv("DATABASE_URL"):
        try:
            from pathlib import Path as _Path
            import asyncio
            import json as _json

            from apps.api.ingest.pipeline_framework import ingest_framework_docx
            from apps.api.ingest.loader import load_canonical_json_to_db
            from apps.api.core.database import get_session

            docx = _Path("docs/source/framework_2025-10_v1.docx")
            if docx.exists():
                # Disable OCR/vector backends during tests unless explicitly enabled.
                os.environ.setdefault("INGEST_OCR_FROM_IMAGES", "off")
                os.environ.setdefault("VECTOR_BACKEND", os.getenv("VECTOR_BACKEND", "disabled"))

                out_dir = _Path("data/derived")
                out_dir.mkdir(parents=True, exist_ok=True)
                canon_path = out_dir / "pytest_framework.canonical.json"
                chunks_path = out_dir / "pytest_framework.chunks.jsonl"

                ingest_framework_docx(docx, canon_path, chunks_path)
                canonical = _json.loads(canon_path.read_text(encoding="utf-8"))

                async def _load():
                    async with get_session() as session:
                        await load_canonical_json_to_db(session, canonical, docx.name)

                asyncio.run(_load())
        except Exception:
            # Tests should still run; DB-dependent ones may skip/fail accordingly.
            pass


@pytest.fixture
def require_db():
    """Skip if database not configured."""
    if not os.getenv("DATABASE_URL"):
        pytest.skip("Requires DATABASE_URL")


@pytest.fixture
def require_llm():
    """Skip if LLM (Azure OpenAI) not configured."""
    if not os.getenv("AZURE_OPENAI_ENDPOINT") or not os.getenv("AZURE_OPENAI_API_KEY"):
        pytest.skip("Requires AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY")


@pytest.fixture
def require_azure_search():
    """Skip if Azure Search not configured."""
    if not os.getenv("AZURE_SEARCH_ENDPOINT") or not os.getenv("AZURE_SEARCH_API_KEY"):
        pytest.skip("Requires AZURE_SEARCH_ENDPOINT and AZURE_SEARCH_API_KEY")
    if os.getenv("VECTOR_BACKEND", "disabled").lower() != "azure_search":
        pytest.skip("Requires VECTOR_BACKEND=azure_search")


@pytest.fixture
def require_db_and_llm():
    """Skip if database or LLM not configured."""
    if not os.getenv("DATABASE_URL"):
        pytest.skip("Requires DATABASE_URL")
    if not os.getenv("AZURE_OPENAI_ENDPOINT") or not os.getenv("AZURE_OPENAI_API_KEY"):
        pytest.skip("Requires AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY")
