"""
Async database session context manager for background tasks
"""
from sqlalchemy.ext.asyncio import AsyncSession
from backend.core.database import async_session_maker
from contextlib import asynccontextmanager


@asynccontextmanager
async def get_db_session():
    """Context manager for standalone database operations"""
    async with async_session_maker() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()