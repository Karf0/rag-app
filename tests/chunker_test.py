from rag_app.embeddings.embedder import Embedder
from rag_app.chunkings.factory import build_chunker


def main():
    text = """For your architecture — clean seams, swappable everything — which violates less? You're trading a redundant load against a component dependency. There's no free answer; both cost something. Tell me which cost you'd rather pay and why, because this is exactly the kind of seam decision that goes in DECISIONS.md. The piece you care about is offset_mapping — a list of (char_start, char_end) tuples, one per token, index-aligned with input_ids. Token i was produced from text[offset_mapping[i][0] : offset_mapping[i][1]].
    Three things to nail down by printing it on a real string, not trusting me:
    1. Special tokens are (0, 0). [CLS]/[SEP] map to no source text, so their offset is a zero-width (0,0). That's your signal to skip them — or pass add_special_tokens=False so they never appear. Either way, don't window over them.
    2. Subword pieces share a word's span. "running" → run + ##ning come back as e.g. (0,3) and (3,7). Two tokens, two offsets, contiguous. Your window slicing on outer bounds handles this for free, but you should see it happen once.
    3. Is char_end exclusive? (0,7) for "running" (7 chars) means end is exclusive — text[0:7] is the word. That makes text[first_start : last_end] a correct Python slice with no +1. Verify this for your tokenizer — it's the one assumption that, if wrong, shifts every chunk boundary by one char.
    One caveat: offset_mapping only appears if the tokenizer is fast (tok.is_fast == True). On a slow tokenizer the key is silently absent. So check that first, or you'll be debugging a missing key.
    Print it on "running fast!" before you write any logic — confirm the shape, the (0,0) specials, and the exclusivity. Then build the loop.Mostly right, but you're conflating two words and the conflation will cost you. Interface and orchestrator are not the same thing — pick the right one or you'll build a confused class.
    An interface (your vector-store seam) is an abstraction with swappable implementations. It exists because you have ≥2 ways to do the thing (pgvector now, Chroma later) and callers must not care which.
    The orchestrator is the opposite: it's a concrete coordinator. There's exactly one. It doesn't abstract anything — it wires together the chunker, embedder, doc store, and vector store and runs the ingest sequence. You don't have two competing orchestrators to hide behind a seam. So it's a concrete class (often called a service), not an interface. Calling it an interface is the mistake.
    Now the real question — is one orchestrator "the only thing you use"? No, and here's the asymmetry you're missing:
    You have two flows, not one:

    Ingest: document → chunk → embed → store both. Writes.
    Query: question → embed → vector search → fetch chunk text → assemble prompt → LLM → answer. Reads.It holds them, it doesn't create them. Dependency injection — the components are constructed elsewhere and handed in. C analogy: the orchestrator is a struct of pointers to subsystems it was given, not the thing that mallocs them. This is exactly the FastAPI dependency-injection mechanic you flagged you haven't learned — it'll click here because the orchestrator is the natural injection target.
    Why DI and not "orchestrator news up its own stores"? Three reasons: you can swap the vector-store impl (your whole migration story) without touching the orchestrator; you can hand it fakes in tests; and you control lifecycle (one embedder loaded once, not re-instantiated per request).
    So, corrected version of your sentence: "For ingest, I have one concrete service that holds the chunker, embedder, and both store interfaces, gets them injected, and runs the sequence. Query is a separate service."
    Two questions before you draft it:"""
    emb = Embedder()
    chunker = build_chunker(emb)
    chunks = chunker.chunk_text(text)
    for i, ch in enumerate(chunks):
        print(f"Chunk {i}, chunk token len {len(chunker.tokenizer.encode(ch))}: \n{ch} ")


if __name__ =="__main__":
    main()