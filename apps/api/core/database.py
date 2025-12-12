"""
Database Connection Module

Provides async database connections using SQLAlchemy.
"""

import os
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase


# Get database URL from environment
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql+asyncpg://wellbeing:wellbeing_dev_password@127.0.0.1:5432/wellbeing_db"
)

# Convert to async URL if needed
if DATABASE_URL.startswith("postgresql://"):
    DATABASE_URL = DATABASE_URL.replace("postgresql://", "postgresql+asyncpg://")

# Windows/localhost IPv6 pitfall:
# On some Windows machines, asyncpg may resolve "localhost" to ::1 and hit a different Postgres listener
# than the IPv4 one used by other clients. Prefer IPv4 unless explicitly disabled.
if os.name == "nt" and os.getenv("DB_HOST_PREFER_IPV4", "true").lower() == "true":
    DATABASE_URL = DATABASE_URL.replace("@localhost:", "@127.0.0.1:")


# Create async engine
engine = create_async_engine(
    DATABASE_URL,
    echo=os.getenv("DEBUG", "").lower() == "true",
    pool_pre_ping=True,
)

# Session factory
async_session_maker = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


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
    async with async_session_maker() as session:
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
    await engine.dispose()

