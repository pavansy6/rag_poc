"""Quick validation script for verifying the saved chunk store.

This script loads a saved chunks pickle file and searches for a sample term
inside the serialized chunk text. It is intended for quick sanity checks after
ingestion or FAISS store creation.
"""

import pickle
chunks_file = 'vectorstore_2026-04-24/chunks.pkl'
try:
    with open(chunks_file, 'rb') as f:
        all_chunks = pickle.load(f)
    print(f'Total chunks indexed: {len(all_chunks)}')
    search_term = 'vault'
    matches = [chunk for chunk in all_chunks if search_term in chunk.lower()]
    if matches:
        print(f"SUCCESS: '{search_term}' exists in index. Found {len(matches)} matches.")
        print('Sample match:', matches[0][:100] + '...' if len(matches[0]) > 100 else matches[0])
    else:
        print(f"FAILURE: '{search_term}' DOES NOT exist in the index.")
        print('This might indicate that document ingestion is incomplete.')
except FileNotFoundError:
    print(f'Error: Could not find {chunks_file}')
    print('Make sure the vector store has been created and saved.')
except Exception as e:
    print(f'Error loading chunks: {e}')