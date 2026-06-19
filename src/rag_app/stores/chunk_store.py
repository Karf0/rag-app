from collections.abc import Sequence
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from rag_app.models.chunk import Chunk
from rag_app.schemas import ChunkDTO


def _to_dto(chunk: Chunk) -> ChunkDTO:
    return ChunkDTO(
        id=chunk.id,
        content=chunk.content,
        document_id=chunk.document_id,
        position=chunk.position,
    )


class ChunkStore:
    def __init__(self, session_factory: async_sessionmaker[AsyncSession]) -> None:
        self._session_factory = session_factory

    async def add_chunks(self, chunks: Sequence[ChunkDTO]) -> None:
        async with self._session_factory() as session:
            session.add_all(
                [ Chunk(id=c.id, content=c.content, document_id=c.document_id, position=c.position,)
                    for c in chunks
                ]
            )
            await session.commit()

    async def get_chunk(self, id: UUID) -> ChunkDTO:
        async with self._session_factory() as session:
            chunk = await session.get(Chunk, id)
            if chunk is None:
                raise ValueError(f"Chunk {id} doesn't exist")
            return _to_dto(chunk)

    async def get_chunks_by_ids(self, ids: Sequence[UUID]) -> list[ChunkDTO]:
        # Text-fetch half of two-step retrieval. Order is not guaranteed here; the caller
        # holds the (chunk_id, distance) ranking from VectorStore.search.
        async with self._session_factory() as session:
            result = await session.execute(select(Chunk).where(Chunk.id.in_(ids)))
            return [_to_dto(c) for c in result.scalars()]

    async def get_chunks_by_document(self, document_id: UUID) -> list[ChunkDTO]:
        async with self._session_factory() as session:
            result = await session.execute(
                select(Chunk)
                .where(Chunk.document_id == document_id)
                .order_by(Chunk.position)
            )
            return [_to_dto(c) for c in result.scalars()]
