"""Text chunking utilities for document ingestion.

This module splits long text documents into overlapping chunks suitable for
vector embedding and retrieval.
"""

from typing import List
from config import CHUNK_SIZE, CHUNK_OVERLAP

class Chunker:

    def chunk(self, text: str) -> List[str]:
        """Chunk a text string into overlapping segments.

        Args:
            text (str): The text to split into chunks.

        Returns:
            List[str]: A list of text chunks, each of size ``CHUNK_SIZE`` with
                ``CHUNK_OVERLAP`` overlap between consecutive chunks.
        """
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