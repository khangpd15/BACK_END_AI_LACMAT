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


def format_async_db_url(url: str) -> str:
    """Ensures DATABASE_URL uses the asyncpg driver and formats SSL parameters properly."""
    if not url:
        return "sqlite+aiosqlite:///:memory:"
    clean = url.strip()
    if clean.startswith("postgres://"):
        clean = clean.replace("postgres://", "postgresql+asyncpg://", 1)
    elif clean.startswith("postgresql://") and not clean.startswith("postgresql+asyncpg://"):
        clean = clean.replace("postgresql://", "postgresql+asyncpg://", 1)

    # asyncpg does not accept 'sslmode' query argument (which libpq / psycopg uses).
    # Instead, asyncpg accepts 'ssl=require' or 'ssl=true'.
    if "sslmode=require" in clean:
        clean = clean.replace("sslmode=require", "ssl=require")
    elif "sslmode=disable" in clean:
        clean = clean.replace("sslmode=disable", "ssl=disable")
    elif "sslmode=" in clean:
        import re
        clean = re.sub(r"sslmode=[^&]+", "ssl=require", clean)
    return clean


def get_engine() -> AsyncEngine:
    """Returns singleton async engine with lazy initialization."""
    global _engine
    if _engine is None:
        raw_url = DATABASE_URL.strip() if DATABASE_URL else ""
        db_url = format_async_db_url(raw_url)
        if "sqlite" in db_url:
            logger.info("DATABASE_URL using SQLite: %s", db_url)
            _engine = create_async_engine(
                db_url,
                echo=False,
                future=True,
            )
        else:
            logger.info("Initializing async PostgreSQL engine...")
            _engine = create_async_engine(
                db_url,
                echo=False,
                future=True,
                pool_pre_ping=True,
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
