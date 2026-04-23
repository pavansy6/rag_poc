"""
Configuration Module

This module stores global settings and prompt configurations for the RAG pipeline.
"""

# The local language model to use for generation (e.g. Qwen for instructions)
# MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"

# SentenceTransformer model used for generating vector embeddings
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Base path to save and load the FAISS vector index
FAISS_INDEX_PATH = "faiss_index"

# Directory where the raw input documents (PDF, TXT, MD) are stored
DOCS_PATH = "docs"

# Defines the chunk size (number of characters/tokens) when splitting documents
CHUNK_SIZE = 500

# Overlap size between consecutive chunks to maintain context
CHUNK_OVERLAP = 50

# SYSTEM_PROMPT = """
# You are a highly experienced Chief Information Security Officer (CISO).
# 
# Your job is to answer questions with:
# - cybersecurity best practices
# - risk-based analysis
# - compliance awareness
# - concise but professional recommendations
# 
# Use the provided context when relevant.
# If the answer is not in the context, rely on your cybersecurity expertise.
# """

# The system prompt that guides the behavior of the LLM generator
SYSTEM_PROMPT = """
You are a highly experienced Chief Information Security Officer (CISO).

Provide detailed, professional, and structured responses.
Use the provided context when relevant.
"""