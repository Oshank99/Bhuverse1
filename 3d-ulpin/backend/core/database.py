"""
Database configuration and session management
"""
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy import text
from contextlib import asynccontextmanager
import os
from typing import AsyncGenerator

from backend.core.models import Base


class DatabaseConfig:
    def __init__(self):
        self.host = os.getenv('POSTGRES_HOST', 'localhost')
        self.port = int(os.getenv('POSTGRES_PORT', '5432'))
        self.database = os.getenv('POSTGRES_DB', 'ulpin3d')
        self.user = os.getenv('POSTGRES_USER', 'postgres')
        self.password = os.getenv('POSTGRES_PASSWORD', 'postgres')
        self.pool_size = int(os.getenv('DB_POOL_SIZE', '10'))
        self.max_overflow = int(os.getenv('DB_MAX_OVERFLOW', '20'))
    
    @property
    def url(self) -> str:
        return f"postgresql+asyncpg://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}"
    
    @property
    def sync_url(self) -> str:
        return f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}"


db_config = DatabaseConfig()

engine = create_async_engine(
    db_config.url,
    pool_size=db_config.pool_size,
    max_overflow=db_config.max_overflow,
    pool_pre_ping=True,
    echo=os.getenv('SQL_ECHO', 'false').lower() == 'true',
)

async_session_maker = async_sessionmaker(
    engine, 
    class_=AsyncSession, 
    expire_on_commit=False,
    autoflush=False
)


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI dependency for database session"""
    async with async_session_maker() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


async def init_database():
    """Initialize database with PostGIS and tables"""
    async with engine.begin() as conn:
        # Enable PostGIS extensions
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS postgis;"))
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS postgis_raster;"))
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS postgis_topology;"))
        
        # Create all tables (migrations handle schema, functions, indexes)
        await conn.run_sync(Base.metadata.create_all)
        
        print("Database initialized with PostGIS 3D support")


async def close_database():
    """Close database connections"""
    await engine.dispose()


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


# Sync engine for Alembic migrations
from sqlalchemy import create_engine
sync_engine = create_engine(db_config.sync_url, echo=False)

def get_sync_session():
    from sqlalchemy.orm import sessionmaker
    SyncSession = sessionmaker(bind=sync_engine)
    return SyncSession()