"""Hybrid retriever combining BM25, vector search, and optional MITRE vector store.

This retriever attempts to blend lexical BM25 results with embedding-based
vector search outputs. For queries that appear to reference MITRE ATT&CK
terminology, and when a dedicated MITRE vector store is available, the
retriever will append additional MITRE-specific chunks to the results.
"""

from typing import List, Callable, Optional, Any


def _extract_text(result: Any) -> str:
    """Normalize a retrieval result to a text string.
    
    Some retrievers return dict-like hits (e.g., {'text': ...}), while
    others may return plain strings. This helper ensures we always work
    with text values.
    
    Args:
        result: The result to extract text from (dict or string).
    
    Returns:
        str: The extracted text content.
    """
    if isinstance(result, dict):
        return result['text']
    return result


class HybridRetriever:
    """Combine multiple retrieval strategies into a single ranked result list.

    Attributes:
        MITRE_SIGNALS (List[str]): Tokens used to heuristically detect MITRE-related queries.
    """
    MITRE_SIGNALS = ['mitre', 'att&ck', 'attack', 'technique', 'tactic', 'threat actor', 'adversar', 'exploit', 'malware', 'ransomware', 'phishing', 'lateral', 'persistence', 'exfiltration', 'command and control', 'c2', 'ttps', 'initial access', 'privilege escalation', 'defense evasion', 'credential', 'discovery', 'collection', 'impact', 'credential dump', 'lsass', 'ntds', 'sam database', 'pass the hash', 'kerberoast', 'mimikatz', 'golden ticket']

    def __init__(self, bm25, vector, mitre_vector=None):
        """Initialize with BM25 and vector retrievers, optionally a MITRE vector store.

        Args:
            bm25: Lexical retriever exposing `retrieve(query, k=...)`.
            vector: Generic vector retriever exposing `retrieve(query)`.
            mitre_vector: Optional specialized vector store for MITRE content with
                a `search(embedding, k=...)` interface.
        """
        self.bm25 = bm25
        self.vector = vector
        self.mitre_vector = mitre_vector

    def _is_mitre_query(self, query: str) -> bool:
        """Heuristically detect whether a query references MITRE ATT&CK concepts.

        Args:
            query (str): The input query text.

        Returns:
            bool: True if any MITRE signal token appears in the lowercased query.
        """
        query_lower = query.lower()
        return any((signal in query_lower for signal in self.MITRE_SIGNALS))

    def retrieve(self, query: str, embed_fn: Optional[Callable[[List[str]], List[List[float]]]]=None) -> List[str]:
        """Retrieve and combine results from BM25 and vector stores, with optional MITRE augmentation.

        The method prefers BM25 and the primary vector store results, de-duplicates
        them while preserving order, and, if the query is MITRE-related and a
        MITRE vector store is available, appends additional MITRE chunks.

        Args:
            query (str): The text query to retrieve for.
            embed_fn (Optional[Callable]): Embedding function required to query
                the MITRE vector store when available.

        Returns:
            List[str]: Up to four combined, de-duplicated text chunks.
        """
        bm25_results = self.bm25.retrieve(query, k=2)
        vector_results = self.vector.retrieve(query)[:1]

        bm25_texts = [_extract_text(r) for r in bm25_results]
        vector_texts = [_extract_text(r) for r in vector_results]
        combined = list(dict.fromkeys(bm25_texts + vector_texts))

        # If a MITRE vector store is provided and the query appears MITRE-related,
        # embed the query and append MITRE-specific chunks (avoiding duplicates).
        if self.mitre_vector and self._is_mitre_query(query) and embed_fn:
            query_embedding = embed_fn([query])[0]
            mitre_results = self.mitre_vector.search(query_embedding, k=4)
            mitre_texts = [hit['text'] for hit in mitre_results]
            for mitre_chunk in mitre_texts:
                if mitre_chunk not in combined:
                    combined.append(mitre_chunk)
        return combined[:4]