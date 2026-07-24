"""Async SQLAlchemy engine/session management.

A `Database` instance wraps one engine + sessionmaker. The app uses the module-level
singleton bound to `settings.database_url`; tests construct their own instance (e.g.
against `sqlite+aiosqlite:///:memory:`) so storage logic is testable without Postgres.
"""
from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker, create_async_engine

from patternlab.config import settings
from patternlab.storage.models import Base


class Database:
    def __init__(self, url: str):
        self.engine: AsyncEngine = create_async_engine(url, pool_pre_ping=True)
        self._session_maker = async_sessionmaker(self.engine, expire_on_commit=False)

    async def init_models(self) -> None:
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    @asynccontextmanager
    async def session(self) -> AsyncIterator[AsyncSession]:
        async with self._session_maker() as session:
            yield session

    async def dispose(self) -> None:
        await self.engine.dispose()


_default_db: Database | None = None


def get_database() -> Database:
    """Lazily-created singleton bound to settings.database_url. Raises if unset."""
    global _default_db
    if not settings.database_url:
        raise RuntimeError("DATABASE_URL is not configured")
    if _default_db is None:
        _default_db = Database(settings.database_url)
    return _default_db
