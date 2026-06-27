"""IngestionService.store_document — the chunk -> embed -> store path (MVP steps 1-4).

Uses the FakeEmbedder/FakeTokenizer doubles so nothing loads SentenceTransformer or hits the
network; the stores, sessions and Chunker are real, against the test DB."""

import re

import pytest
from sqlalchemy.ext.asyncio import async_sessionmaker
from uuid import uuid4

from rag_app.chunkings.chunker import Chunker
from rag_app.schemas import DocumentDTO
from rag_app.services.ingestor import IngestionService


# 40 whitespace-tokens; with max_size=20 / overlap=5 the FakeTokenizer windowing yields 3 chunks
# (same arithmetic as test_chunking.test_chunker_1), so the stored chunk/vector count is known.
_FORTY_WORDS = """w01 w02 w03 w04 w05 w06 w07 w08 w09 w10 w11 w12 w13 w14 w15 w16 w17 w18 w19 w20 w21 w22 w23 w24 w25 w26 w27 w28 w29 w30 w31 w32 w33 w34 w35 w36 w37 w38 w39 w40"""
_EXPECTED_CHUNKS = 3


def _make_ingestor(engine, doc_store, chunk_store, vector_store, fake_embedder, fake_tokenizer):
    chunker = Chunker(fake_tokenizer, 20, 5)
    session_factory = async_sessionmaker(engine, expire_on_commit=False)
    return IngestionService(doc_store, chunk_store, vector_store, fake_embedder, chunker, session_factory)


async def test_store_document_persists_doc_chunks_vectors(
    engine, doc_store, chunk_store, vector_store, fake_embedder, fake_tokenizer, new_session, tmp_path, db_tests
):
    f = tmp_path / "doc.txt"
    f.write_text(_FORTY_WORDS)
    doc = DocumentDTO(uuid4(), "doc.txt", str(f), "hash-it", {"creator": "ambulance"})
    ingestor = _make_ingestor(engine, doc_store, chunk_store, vector_store, fake_embedder, fake_tokenizer)

    await ingestor.store_document(doc)

    async with new_session() as s:
        ret_doc = await doc_store.get_document(s, doc.id)
        chunks = await chunk_store.get_chunks_by_document(s, doc.id)
        vectors = [await vector_store.get_vector_values_by_chunk_id(s, ch.id) for ch in chunks]

    assert ret_doc.id == doc.id
    assert ret_doc.content_hash == doc.content_hash
    assert len(chunks) == _EXPECTED_CHUNKS
    assert [ch.position for ch in chunks] == list(range(_EXPECTED_CHUNKS))
    # One vector per chunk, each the configured pgvector dimension.
    assert len(vectors) == _EXPECTED_CHUNKS
    assert all(len(v) == fake_embedder.dimension for v in vectors)


async def test_store_document_dedupes_on_content_hash(
    engine, doc_store, chunk_store, vector_store, fake_embedder, fake_tokenizer, new_session, tmp_path, db_tests
):
    # Two distinct documents (different ids) sharing a content_hash: the second is a no-op because
    # DocStore.exists matches on content_hash.
    f1 = tmp_path / "first.txt"
    f1.write_text(_FORTY_WORDS)
    f2 = tmp_path / "second.txt"
    f2.write_text(_FORTY_WORDS)
    doc1 = DocumentDTO(uuid4(), "first.txt", str(f1), "same-hash", {})
    doc2 = DocumentDTO(uuid4(), "second.txt", str(f2), "same-hash", {})
    ingestor = _make_ingestor(engine, doc_store, chunk_store, vector_store, fake_embedder, fake_tokenizer)

    await ingestor.store_document(doc1)
    await ingestor.store_document(doc2)

    async with new_session() as s:
        assert (await doc_store.get_document(s, doc1.id)).id == doc1.id
        with pytest.raises(ValueError):
            await doc_store.get_document(s, doc2.id)
        # Only doc1's chunks exist; doc2 never got past the dedupe short-circuit.
        assert len(await chunk_store.get_chunks_by_document(s, doc1.id)) == _EXPECTED_CHUNKS


async def test_store_document_rejects_whitespace_only(
    engine, doc_store, chunk_store, vector_store, fake_embedder, fake_tokenizer, tmp_path, db_tests
):
    f = tmp_path / "blank.txt"
    f.write_text("   \n\t \v \n")
    doc = DocumentDTO(uuid4(), "blank.txt", str(f), "hash-blank", {})
    ingestor = _make_ingestor(engine, doc_store, chunk_store, vector_store, fake_embedder, fake_tokenizer)

    with pytest.raises(ValueError):
        await ingestor.store_document(doc)


async def test_store_document_rejects_missing_path(
    engine, doc_store, chunk_store, vector_store, fake_embedder, fake_tokenizer, tmp_path, db_tests
):
    missing = tmp_path / "nope.txt"  # never created
    doc = DocumentDTO(uuid4(), "nope.txt", str(missing), "hash-missing", {})
    ingestor = _make_ingestor(engine, doc_store, chunk_store, vector_store, fake_embedder, fake_tokenizer)

    with pytest.raises(ValueError):
        await ingestor.store_document(doc)
