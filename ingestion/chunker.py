from config import CHUNK_SIZE, CHUNK_OVERLAP

class Chunker:
    """
    A utility class to split large text strings into smaller, manageable chunks 
    with a defined size and overlap for vector ingestion.
    """
    def chunk(self, text):
        """
        Splits the input text into a list of chunks based on CHUNK_SIZE and CHUNK_OVERLAP.
        
        Args:
            text (str): The raw text extracted from documents.
        
        Returns:
            list[str]: A list of text chunks.
        """
        chunks = []
        start = 0

        while start < len(text):
            end = start + CHUNK_SIZE
            chunks.append(text[start:end])
            # Slide the window forward, keeping some overlap to to avoid cutting off mid-sentence context
            start += CHUNK_SIZE - CHUNK_OVERLAP

        return chunks