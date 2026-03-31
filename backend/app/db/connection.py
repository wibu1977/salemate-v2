import asyncpg
from contextlib import asynccontextmanager
from app.config import get_settings
from typing import AsyncGenerator, Callable, Any
from functools import wraps


class DatabasePool:
    """Async PostgreSQL connection pool."""

    _pool: asyncpg.Pool | None = None

    @classmethod
    async def init(cls) -> None:
        """Initialize connection pool."""
        if cls._pool is None:
            settings = get_settings()
            cls._pool = await asyncpg.create_pool(
                settings.supabase_url.replace("postgresql://", "postgresql://postgres:"),
                min_size=5,
                max_size=20,
                command_timeout=60,
            )

    @classmethod
    async def close(cls) -> None:
        """Close all connections."""
        if cls._pool:
            await cls._pool.close()
            cls._pool = None

    @classmethod
    @asynccontextmanager
    async def acquire(cls) -> AsyncGenerator[asyncpg.Connection, None]:
        """Yield a connection from the pool."""
        if cls._pool is None:
            await cls.init()
        async with cls._pool.acquire() as conn:
            yield conn

    @classmethod
    def pool(cls) -> asyncpg.Pool:
        """Get the pool (for type checking)."""
        if cls._pool is None:
            raise RuntimeError("Pool not initialized")
        return cls._pool


# For testing: allow overriding get_db
_get_db_override: Callable[[], Any] = None


def override_get_db(override: Callable[[], Any]) -> None:
    """Override get_db for testing."""
    global _get_db_override
    _get_db_override = override


def reset_get_db_override() -> None:
    """Reset get_db override."""
    global _get_db_override
    _get_db_override = None


async def get_db() -> AsyncGenerator[asyncpg.Connection, None]:
    """FastAPI dependency for database connection."""
    global _get_db_override
    if _get_db_override:
        async for conn in _get_db_override():
            yield conn
    else:
        async with DatabasePool.acquire() as conn:
            yield conn
