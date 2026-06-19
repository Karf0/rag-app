from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.orm import Mapped
from sqlalchemy.orm import mapped_column
from sqlalchemy.orm import relationship
from sqlalchemy import ForeignKey
from pgvector.sqlalchemy import Vector as PgVector
from pathlib import Path
from uuid import UUID


class Base(DeclarativeBase):
    pass

class Document(Base):
    __tablename__ = "stored_documents"
    
    id: Mapped[UUID] = mapped_column(primary_key=True)
    filename: Mapped[str]
    path_raw_content: Mapped[str]
    chunks: Mapped[list["Chunk"]] = relationship(back_populates="original_document")

    def __repr__(self) -> str:
        return f"Document(id={self.id!r}, filename={self.filename!r}, raw_content={self.path_raw_content!r})"

class Chunk(Base):
    __tablename__ = "stored_chunks"

    id: Mapped[UUID] = mapped_column(primary_key=True)
    content: Mapped[str]
    document_id: Mapped[UUID] = mapped_column(ForeignKey("stored_documents.id"))

    original_document: Mapped[Document] = relationship(back_populates="chunks")
    vector: Mapped["Vector"] = relationship(back_populates="original_chunk")
    def __repr__(self) -> str:
        return f"Chunk(id={self.id}, content={self.content})"
    

class Vector(Base):
    __tablename__ = "stored_vectors"

    chunk_id: Mapped[UUID] = mapped_column(ForeignKey("stored_chunks.id"), primary_key=True)
    content: Mapped[list[float]] = mapped_column(PgVector(384))

    original_chunk: Mapped[Chunk] = relationship(back_populates="vector")

