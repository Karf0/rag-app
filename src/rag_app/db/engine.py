from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from rag_app.config import get_settings

_url = get_settings().sqlalchemy_url

engine = create_async_engine(_url)

# expire_on_commit=False: in async we must avoid implicit lazy-load I/O triggered by
# attribute access after commit (a refresh would emit SQL outside an awaited call).
AsyncSessionLocal: async_sessionmaker[AsyncSession] = async_sessionmaker(
    engine, expire_on_commit=False
)
