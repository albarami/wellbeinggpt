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
                    # Don't override existing env vars unless they are empty.
                    # Reason: in some environments keys may exist but be blank, which previously
                    # caused .env loading to silently do nothing and made LLM tests skip.
                    if key and (key not in os.environ or not (os.environ.get(key) or "").strip()):
                        os.environ[key] = value
        
        # Always enable DB tests when .env is loaded
        os.environ.setdefault("RUN_DB_TESTS", "1")

        # If the user has no Azure Search, use local BM25 for "vector" retrieval in tests.
        # This is not mock data; it ranks real chunks already in Postgres.
        if os.getenv("VECTOR_BACKEND", "disabled").lower() == "disabled":
            os.environ["VECTOR_BACKEND"] = "bm25"
        
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
            from apps.api.ingest.chunk_span_store import populate_chunk_spans_for_source
            from apps.api.ingest.scholar_notes_loader import ingest_scholar_notes_jsonl
            from apps.api.core.database import get_session
            from apps.api.core.schema_bootstrap import bootstrap_db

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
                    # Ensure schema matches current code (adds new columns/tables deterministically).
                    await bootstrap_db()
                    async with get_session() as session:
                        summary = await load_canonical_json_to_db(session, canonical, docx.name)
                        # Best-effort: persist deterministic spans for stable citations.
                        try:
                            sd = str((summary or {}).get("source_doc_id") or "")
                            if sd:
                                await populate_chunk_spans_for_source(session, sd)
                        except Exception:
                            pass

                        # Commit framework ingestion before generating scholar notes.
                        # Reason: note generation opens a new session and must see chunks.
                        await session.commit()

                        # Ingest scholar notes pack if present (required for deep-mode evaluation).
                        notes_path = _Path("data/scholar_notes/notes_v1.jsonl")
                        if notes_path.exists() and notes_path.stat().st_size > 0:
                            # Regenerate a deterministic starter pack from the current DB so chunk_ids match.
                            from scripts.generate_scholar_notes_v1 import _run as _gen_notes  # type: ignore

                            await _gen_notes(out_path=notes_path, limit=12, version="v1")
                            await ingest_scholar_notes_jsonl(
                                session=session,
                                notes_jsonl_path=str(notes_path),
                                pack_name="scholar_notes_v1",
                            )

                        await session.commit()

                asyncio.run(_load())
        except Exception:
            # Reason: RUN_DB_TESTS=1 is an explicit opt-in; failures should be loud.
            raise


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
    if not os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME"):
        pytest.skip("Requires AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME")


@pytest.fixture
def require_db_and_llm():
    """Skip if database or LLM not configured."""
    if not os.getenv("DATABASE_URL"):
        pytest.skip("Requires DATABASE_URL")
    if not os.getenv("AZURE_OPENAI_ENDPOINT") or not os.getenv("AZURE_OPENAI_API_KEY"):
        pytest.skip("Requires AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY")
