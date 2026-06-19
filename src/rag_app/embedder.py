from sentence_transformers import SentenceTransformer


class Embedder:
    def __init__(self):
        self.model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    def embed_document(self, texts: list[str]) -> list[list[float]]:
        return self.model.encode_document(texts).tolist()
    def embed_query(self, query: str) -> list[list[float]]:
        return self.model.encode_query(query).tolist()
    




emb = Embedder()
ls = emb.embed_document([
    "The weather is lovely today.",
    "It's so sunny outside!",
    "He drove to the stadium.", ])
print(ls)