from typing import List
from config import CHUNK_SIZE, CHUNK_OVERLAP

class Chunker:

    def chunk(self, text: str) -> List[str]:
        if not text:
            return []
        chunks = []
        start_position = 0
        while start_position < len(text):
            end_position = start_position + CHUNK_SIZE
            chunk = text[start_position:end_position]
            chunks.append(chunk)
            start_position += CHUNK_SIZE - CHUNK_OVERLAP
            if start_position >= len(text):
                break
        return chunks