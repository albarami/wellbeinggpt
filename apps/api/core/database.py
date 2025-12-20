"""
Database Connection Module

Provides async database connections using SQLAlchemy.
"""

import os
import sys
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.pool import NullPool


def _db_url_from_env() -> str:
    """Compute DB URL from current environment."""
    url = os.getenv(
        "DATABASE_URL",
        "postgresql+asyncpg://wellbeing:wellbeing_dev_password@127.0.0.1:5432/wellbeing_db",
    )
    if url.startswith("postgresql://"):
        url = url.replace("postgresql://", "postgresql+asyncpg://")
    # Windows/localhost IPv6 pitfall: prefer IPv4.
    if os.name == "nt" and os.getenv("DB_HOST_PREFER_IPV4", "true").lower() == "true":
        url = url.replace("@localhost:", "@127.0.0.1:")
    return url


_engine = None
_engine_url: str | None = None
_session_maker = None


def _get_engine():
    """Get (or create) an engine bound to the current env DATABASE_URL."""
    global _engine, _engine_url, _session_maker
    url = _db_url_from_env()

    # Reason: some CLI tools load .env after import; ensure engine reflects env.
    if _engine is None or _engine_url != url:
        use_null_pool = "pytest" in sys.modules
        _engine = create_async_engine(
            url,
            echo=os.getenv("DEBUG", "").lower() == "true",
            pool_pre_ping=True,
            poolclass=NullPool if use_null_pool else None,
        )
        _engine_url = url
        _session_maker = async_sessionmaker(
            _engine,
            class_=AsyncSession,
            expire_on_commit=False,
        )
    return _engine


def _get_session_maker():
    _get_engine()
    return _session_maker


class Base(DeclarativeBase):
    """Base class for SQLAlchemy models."""

    pass


@asynccontextmanager
async def get_session() -> AsyncGenerator[AsyncSession, None]:
    """
    Get an async database session.

    Usage:
        async with get_session() as session:
            # use session

    Yields:
        AsyncSession: Database session.
    """
    sm = _get_session_maker()
    async with sm() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


async def init_db() -> None:
    """
    Initialize database (create tables).

    This is a placeholder - in production, use migrations.
    """
    # Tables are created via schema.sql or migrations
    pass


async def close_db() -> None:
    """Close database connections."""
    global _engine
    if _engine is not None:
        await _engine.dispose()
        _engine = None

