"""Database connection and session management for RemiCare Cover Test."""

from contextlib import asynccontextmanager
import logging
from typing import AsyncGenerator, Optional
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase
from app.config import DATABASE_URL

logger = logging.getLogger("remicare.db")


class Base(DeclarativeBase):
    """Base declarative class for all database models."""
    pass


_engine: Optional[AsyncEngine] = None
_session_factory: Optional[async_sessionmaker[AsyncSession]] = None


def get_engine() -> AsyncEngine:
    """Returns singleton async engine with lazy initialization."""
    global _engine
    if _engine is None:
        db_url = DATABASE_URL.strip() if DATABASE_URL else ""
        if not db_url:
            # Fallback to in-memory async SQLite for offline development and unit tests
            db_url = "sqlite+aiosqlite:///:memory:"
            logger.info("DATABASE_URL not set; using in-memory SQLite: %s", db_url)
        _engine = create_async_engine(
            db_url,
            echo=False,
            future=True,
        )
    return _engine


def get_session_factory() -> async_sessionmaker[AsyncSession]:
    """Returns singleton sessionmaker."""
    global _session_factory
    if _session_factory is None:
        engine = get_engine()
        _session_factory = async_sessionmaker(
            bind=engine,
            class_=AsyncSession,
            expire_on_commit=False,
        )
    return _session_factory


async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI dependency for yielding database session."""
    factory = get_session_factory()
    async with factory() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise


async def init_db() -> None:
    """Initializes tables in database if they do not exist."""
    engine = get_engine()
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info("Database tables initialized successfully.")
