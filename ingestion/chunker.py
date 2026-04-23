from config import CHUNK_SIZE, CHUNK_OVERLAP

class Chunker:
    def chunk(self, text):
        chunks = []
        start = 0

        while start < len(text):
            end = start + CHUNK_SIZE
            chunks.append(text[start:end])
            start += CHUNK_SIZE - CHUNK_OVERLAP

        return chunks