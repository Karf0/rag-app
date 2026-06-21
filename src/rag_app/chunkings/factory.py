from rag_app.chunkings.chunker import Chunker
from rag_app.config import Settings, get_settings
from rag_app.embeddings.embedder import Embedder


def build_chunker(embedder: Embedder, settings: Settings | None = None) -> Chunker:
    """Wire a Chunker to the embedder's tokenizer and token budget.

    This is the single place that knows both the configured chunk_size and the model's actual
    window, so the size guard lives here: a chunk larger than max_content_tokens would be
    silently truncated at embed time, breaking text<->vector agreement. Keeping this out of the
    Chunker class leaves Chunker decoupled from Embedder.
    """
    settings = settings or get_settings()
    ceiling = embedder.max_content_tokens

    size = settings.chunk_size if settings.chunk_size is not None else ceiling
    if size > ceiling:
        raise ValueError(
            f"chunk_size {size} exceeds the model's content window {ceiling}; "
            f"chunks would be truncated at embedding time."
        )

    overlap = round(size * settings.chunk_overlap_ratio)
    return Chunker(embedder.tokenizer, size, overlap)
