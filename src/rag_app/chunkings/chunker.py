from rag_app.schemas import ChunkDTO

from rag_app.config import get_settings

class Chunker:
    def __init__(self, tokenizer, max_size: int, overlap: int) -> None:
        if max_size > get_settings().max_chunk_size:
            raise ValueError(f"max_size can't be bigger than the model's max_size: {get_settings().max_chunk_size}")
        if max_size <= 0 or overlap < 0:
            raise ValueError("max_size and size have to be >= 0")
        if overlap >= max_size:
            raise ValueError("overlap can't be >= than max_size")
        self.tokenizer = tokenizer
        self.max_size = max_size
        self.overlap = overlap
    def chunk_text(self, text: str) -> list[str]:
        out = self.tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
        offsets = out["offset_mapping"]
        # guarded also at ingestor with checking for empty doc, but chunker itself has to be safe
        if not offsets:
            return []
        char_start = char_end = 0
        tok_start = tok_end = 0
        ret: list[str] = []
        while True:
            if tok_start + self.max_size >= len(offsets):
                char_start = offsets[tok_start][0]
                ret.append(text[char_start:])
                return ret
            
            tok_end = tok_start + self.max_size
            char_start = offsets[tok_start][0]
            char_end = offsets[tok_end-1][1]
            ret.append(text[char_start:char_end])
            tok_start = tok_end - self.overlap
            


