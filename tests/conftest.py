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
    else:
        print(f"\n[conftest] No .env file found at: {env_file}")


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


@pytest.fixture
def require_db_and_llm():
    """Skip if database or LLM not configured."""
    if not os.getenv("DATABASE_URL"):
        pytest.skip("Requires DATABASE_URL")
    if not os.getenv("AZURE_OPENAI_ENDPOINT") or not os.getenv("AZURE_OPENAI_API_KEY"):
        pytest.skip("Requires AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY")
