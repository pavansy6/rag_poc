"""Quick validation script for verifying the RAG pipeline and vector stores.

This script tests the document and MITRE stores by loading them and verifying
that they contain content. It is intended for quick sanity checks.
"""

import os
from config import FAISS_INDEX_PATH, MITRE_INDEX_PATH
from rag.pipeline_builder import build_document_store, build_mitre_store


def test_document_store():
    """Test document store loading and basic functionality."""
    print("Testing document store...")
    try:
        store, chunks = build_document_store()
        print(f"✓ Document store loaded: {len(chunks)} chunks indexed")
        if chunks:
            print(f"  Sample chunk (first 100 chars): {chunks[0][:100]}...")
        return True
    except Exception as e:
        print(f"✗ Document store test failed: {e}")
        return False


def test_mitre_store():
    """Test MITRE store loading and basic functionality."""
    print("Testing MITRE store...")
    try:
        mitre_store = build_mitre_store()
        if mitre_store and len(mitre_store.texts) > 0:
            print(f"✓ MITRE store loaded: {len(mitre_store.texts)} chunks indexed")
            print(f"  Sample chunk (first 100 chars): {mitre_store.texts[0][:100]}...")
            return True
        else:
            print("✗ MITRE store is empty")
            return False
    except Exception as e:
        print(f"✗ MITRE store test failed: {e}")
        return False


if __name__ == '__main__':
    print("=" * 60)
    print("RAG Pipeline Validation Tests")
    print("=" * 60)
    
    doc_ok = test_document_store()
    print()
    mitre_ok = test_mitre_store()
    
    print()
    print("=" * 60)
    if doc_ok and mitre_ok:
        print("✓ All tests passed!")
    elif doc_ok:
        print("⚠ Document store OK, but MITRE store failed or is disabled")
    else:
        print("✗ Tests failed - check configuration and file paths")
    print("=" * 60)