import base
import asyncio
import os
import aiofiles
from os.path import isfile
from uuid import UUID
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.ext.asyncio import AsyncSession

class DocStore:
    def __init__(self):
        self.engine = create_async_engine(f"postgresql+asyncpg://postgres:{os.environ['DB_PASSWORD']}@localhost:5432/ragdb")
    
    async def add_document(self, document: base.Document) -> None:
        async with AsyncSession(self.engine) as session:
            session.add(document)
            await session.commit()
    
    async def get_document(self, id: UUID) -> base.Document:
        async with AsyncSession(self.engine) as session:
            doc = await session.get(base.Document, id)
            if doc is None:
                raise ValueError("Document doesn't exist")
            # returning of dead ORM objects - the loaded attributes are OK, the lazy are not it will try to execute
            # SQL on a closed session -> error
            # not return ORM objects at all
            return doc 
        
    async def get_document_content(self, id: UUID) -> str:
        doc = await self.get_document(id)
        if not isfile(doc.path_raw_content):
            raise OSError(f"File at {doc.path_raw_content!r} doesn't exist")
        async with aiofiles.open(doc.path_raw_content, mode="r") as f:
            return await f.read()
    # no cascade - ORM or DB = error when chunks without doc_id
    async def remove_document(self, id: UUID) -> None:
        async with AsyncSession(self.engine) as session:
            to_del = await session.get(base.Document, id)
            if not to_del:
                raise ValueError("Document doesn't exist")
            await session.delete(to_del)
            await session.commit()
    