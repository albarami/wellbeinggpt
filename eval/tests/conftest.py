"""Eval test configuration.

We re-export fixtures defined in tests/conftest.py so they are available under
this test subtree as well.

Additionally, when running only eval tests (without collecting tests/),
pytest won't execute tests/conftest.py's pytest_configure. We therefore
best-effort load .env here as well (guarded).
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest


def pytest_configure(config) -> None:
    # Guard: if the environment is already configured, don't re-run.
    if os.getenv("DATABASE_URL") or os.getenv("EVAL_DOTENV_LOADED") == "1":
        return

    env_file = Path(__file__).resolve().parents[2] / ".env"
    if env_file.exists():
        with open(env_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" in line:
                    key, _, value = line.partition("=")
                    key = key.strip()
                    value = value.strip()
                    if key and key not in os.environ:
                        os.environ[key] = value

        os.environ.setdefault("RUN_DB_TESTS", "1")
        if os.getenv("VECTOR_BACKEND", "disabled").lower() == "disabled":
            os.environ["VECTOR_BACKEND"] = "bm25"

    os.environ["EVAL_DOTENV_LOADED"] = "1"


from tests.conftest import (  # noqa: E402,F401
    require_db,
    require_llm,
    require_db_and_llm,
    require_azure_search,
)


@pytest.fixture
def require_db_strict() -> None:
    """
    Require DATABASE_URL for eval gates; do not skip.

    Reason: evaluation harness is a non-negotiable gate. If DB isn't configured,
    the correct behavior is a hard failure, not a skip.
    """
    if not os.getenv("DATABASE_URL"):
        raise RuntimeError("DATABASE_URL is required for eval tests (no skips allowed).")
