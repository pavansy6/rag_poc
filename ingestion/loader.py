import os
from pypdf import PdfReader
from mitre_chunker import load_mitre_documents   # drop mitre_chunker.py in project root


class Loader:
    """
    Loads documents from disk (txt / md / pdf) and MITRE ATT&CK JSON.
    Existing load_documents() is untouched — load_mitre() is a new addition.
    """

    def load_documents(self, folder):
        """
        Iterates over a directory and parses the contents of text-based and PDF files.

        Args:
            folder (str): The path to the folder containing documents to process.

        Returns:
            list[str]: A list of extracted text strings, one per file.
        """
        docs = []

        for file in os.listdir(folder):
            path = os.path.join(folder, file)

            if file.endswith(".txt") or file.endswith(".md"):
                with open(path, "r", encoding="utf-8") as f:
                    docs.append(f.read())

            elif file.endswith(".pdf"):
                reader = PdfReader(path)
                text = "\n".join(
                    [page.extract_text() for page in reader.pages if page.extract_text()]
                )
                docs.append(text)

        return docs

    def load_mitre(self, json_path):
        """
        Parses the MITRE ATT&CK enterprise JSON and returns structured chunks
        ready for embedding and ingestion into FAISSStore.

        Args:
            json_path (str): Path to enterprise-attack.json

        Returns:
            tuple[list[str], list[dict]]:
                - texts:    page_content strings to embed
                - metadata: parallel list of metadata dicts (chunk_type, tactic_id, etc.)
        """
        docs = load_mitre_documents(json_path)
        texts    = [doc.page_content for doc in docs]
        metadata = [doc.metadata     for doc in docs]
        return texts, metadata