import base
import os
from uuid import UUID
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from pgvector.sqlalchemy import Vector as PgVector


class VectorStore:
    # 2 different pools of connections (doc_store) - not good
    def __init__(self):
        self.engine = create_async_engine(f"postgresql+asyncpg://postgres:{os.environ['DB_PASSWORD']}@localhost:5432/ragdb")

    async def add_vector(self, vector: list[float], original_chunk_id: UUID) -> None:
        async with AsyncSession(self.engine) as session:
            to_add = base.Vector(content=vector, chunk_id=original_chunk_id)
            session.add(to_add)
            await session.commit()
    
    # returning ORM entities - back-end leaking - bad 
    async def get_vector_by_chunk_id(self, original_chunk_id: UUID) -> base.Vector:
        async with AsyncSession(self.engine) as session:
            result = await session.execute(
                select(base.Vector).where(base.Vector.chunk_id==original_chunk_id) 
            )
            vector = result.scalar_one_or_none()
            if vector is None:
                raise ValueError(f"Vector with {original_chunk_id} doesn't exist")
            return vector
        
    async def get_vector_values_by_chunk_id(self, original_chunk_id: UUID) -> list[float]:
        vector = await self.get_vector_by_chunk_id(original_chunk_id)
        return vector.content

    async def search(self, query_vector: list[float], k: int) -> list[tuple[UUID, float]]:
        if k <= 0:
            raise ValueError("k has to be >= 1")
        async with AsyncSession(self.engine) as session:
            distance = base.Vector.content.cosine_distance(query_vector)
            stmt = select(base.Vector.chunk_id, distance).order_by(distance).limit(k)
            result = await session.execute(stmt)
            return [(ret_id, ret_dist) for ret_id, ret_dist in result]
    