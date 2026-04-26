import os
from pypdf import PdfReader

class Loader:
    """
    A utility class to read and extract text from various document formats 
    (txt, md, pdf) within a specified directory.
    """
    def load_documents(self, folder):
        """
        Iterates over a directory and parses the contents of text-based and PDF files.
        
        Args:
            folder (str): The path to the folder containing documents to process.
        
        Returns:
            list[str]: A list of extracted text strings, where each element represents the content of a file.
        """
        docs = []

        for file in os.listdir(folder):
            path = os.path.join(folder, file)

            # Load flat text files
            if file.endswith(".txt") or file.endswith(".md"):
                with open(path, "r", encoding="utf-8") as f:
                    docs.append(f.read())

            # Load PDF files by extracting text page by page
            elif file.endswith(".pdf"):
                reader = PdfReader(path)
                text = "\n".join(
                    [page.extract_text() for page in reader.pages if page.extract_text()]
                )
                docs.append(text)

        return docs