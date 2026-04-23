class Chunker:
    def chunk(self, text):
        paragraphs = text.split("\n\n")
        chunks = []

        current_chunk = ""

        for para in paragraphs:
            if len(current_chunk) + len(para) < 1000:
                current_chunk += "\n\n" + para
            else:
                chunks.append(current_chunk.strip())
                current_chunk = para

        if current_chunk:
            chunks.append(current_chunk.strip())

        return chunks