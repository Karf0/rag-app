def build_prompt(query: str, chunks: list[str]) -> str:
    context = "\n---\n".join(chunks)
    return (
        "Instructions:\n"
        " - Answer the question referencing ONLY the context below for facts.\n"
        " - If the context doesn't contain the answer, say you can't give a confident answer.\n"
        " - The '---' is used for dividing Context Chunks - after each '---' comes new context chunk for you to use.\n"
        "Context:\n"
        f"{context}\n\n"
        "Question:\n"
        f"{query}\n\n"
        "Answer:"
    )
