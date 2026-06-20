from os.path import isfile
from uuid import UUID

import aiofiles
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from rag_app.models.document import Document
from rag_app.schemas import DocumentDTO


class DocStore:
    def __init__(self, session_factory: async_sessionmaker[AsyncSession]) -> None:
        self._session_factory = session_factory

    async def add_document(self, document: DocumentDTO) -> None:
        async with self._session_factory() as session:
            session.add(
                Document(
                    id=document.id,
                    filename=document.filename,
                    path_raw_content=document.path_raw_content,
                    doc_metadata=document.doc_metadata,
                )
            )
            await session.commit()

    async def get_document(self, id: UUID) -> DocumentDTO:
        async with self._session_factory() as session:
            doc = await session.get(Document, id)
            if doc is None:
                raise ValueError(f"Document {id} doesn't exist")
            return DocumentDTO(
                id=doc.id,
                filename=doc.filename,
                path_raw_content=doc.path_raw_content,
                doc_metadata=doc.doc_metadata,
            )

    async def get_document_content(self, id: UUID) -> str:
        doc = await self.get_document(id)
        if not isfile(doc.path_raw_content):
            raise OSError(f"File at {doc.path_raw_content!r} doesn't exist")
        async with aiofiles.open(doc.path_raw_content, mode="r") as f:
            return await f.read()

    async def remove_document(self, id: UUID) -> None:
        async with self._session_factory() as session:
            doc = await session.get(Document, id)
            if doc is None:
                raise ValueError(f"Document {id} doesn't exist")
            await session.delete(doc)
            await session.commit()

    async def exists(self, id: UUID) -> bool:
        async with self._session_factory() as session:
            doc = await session.get(Document, id)
            if doc is None:
                return False
            return True
