import os
from pypdf import PdfReader

class Loader:
    def load_documents(self, folder):
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