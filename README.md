# Local RAG Assistant

This project is a simple in-house RAG (Retrieval-Augmented Generation) assistant built in Python.

It reads internal documents, converts them into embeddings, stores them in a local FAISS vector database, and uses a local Hugging Face model to answer questions based on those documents.

The current setup is focused on cybersecurity and compliance-related questions, and the model is guided to answer like a CISO.

---

## How it works

The flow is simple:

1. Documents are placed inside the `docs/` folder.

2. The loader reads supported files such as:

   * `.txt`
   * `.md`
   * `.pdf`

3. Large documents are split into smaller chunks.

4. Each chunk is converted into embeddings using a sentence transformer model.

5. The embeddings are stored in FAISS for fast similarity search.

6. When a user asks a question:

   * the question is converted into an embedding
   * FAISS finds the most relevant chunks
   * those chunks are added as context in the prompt
   * the LLM generates the final answer

---

## Folder Structure

```bash
rag_poc/
│
├── app.py                  # terminal version
├── streamlit_app.py        # web UI version
├── config.py               # configs and model names
├── faiss_index.index       # saved FAISS vector index
├── faiss_index.pkl         # stored text chunks / metadata
│
├── docs/                   # knowledge base documents
│
├── models/
│   ├── llm.py              # loads Hugging Face model
│   └── embeddings.py       # embedding model
│
├── ingestion/
│   ├── loader.py           # loads files
│   └── chunker.py          # splits text into chunks
│
├── vectordb/
│   └── faiss_store.py      # FAISS logic
│
├── retrieval/
│   └── retriever.py        # retrieves relevant chunks
│
└── rag/
    └── engine.py           # builds prompt and gets answer
```

---

## Installation

Install dependencies:

```bash
uv sync
```

Or manually:

```bash
uv add transformers torch torchvision sentence-transformers faiss-cpu streamlit pypdf
```

---

## Run in Terminal

```bash
uv run app.py
```

Example:

```bash
Ask: What is the patching SLA for critical vulnerabilities?
```

---

## Run in Browser

```bash
uv run streamlit run streamlit_app.py
```

Then open:

```bash
http://localhost:8501
```

---

## Notes

* The first run may take some time because models are downloaded from Hugging Face.
* Downloaded models are cached locally.
* FAISS files are stored in the root directory.
* If you add new documents, re-run the app to rebuild embeddings.

---

