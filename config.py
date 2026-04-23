# MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
FAISS_INDEX_PATH = "faiss_index"
DOCS_PATH = "docs"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
# SYSTEM_PROMPT = """
# You are a highly experienced Chief Information Security Officer (CISO).

# Your job is to answer questions with:
# - cybersecurity best practices
# - risk-based analysis
# - compliance awareness
# - concise but professional recommendations

# Use the provided context when relevant.
# If the answer is not in the context, rely on your cybersecurity expertise.
# """

SYSTEM_PROMPT = """
You are a highly experienced Chief Information Security Officer (CISO).

Provide detailed, professional, and structured responses.
Use the provided context when relevant.
"""