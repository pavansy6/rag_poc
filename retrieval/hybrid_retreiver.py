from typing import List, Callable, Optional, Any

class HybridRetriever:
    MITRE_SIGNALS = ['mitre', 'att&ck', 'attack', 'technique', 'tactic', 'threat actor', 'adversar', 'exploit', 'malware', 'ransomware', 'phishing', 'lateral', 'persistence', 'exfiltration', 'command and control', 'c2', 'ttps', 'initial access', 'privilege escalation', 'defense evasion', 'credential', 'discovery', 'collection', 'impact', 'credential dump', 'lsass', 'ntds', 'sam database', 'pass the hash', 'kerberoast', 'mimikatz', 'golden ticket']

    def __init__(self, bm25, vector, mitre_vector=None):
        self.bm25 = bm25
        self.vector = vector
        self.mitre_vector = mitre_vector

    def _is_mitre_query(self, query: str) -> bool:
        query_lower = query.lower()
        return any((signal in query_lower for signal in self.MITRE_SIGNALS))

    def retrieve(self, query: str, embed_fn: Optional[Callable[[List[str]], List[List[float]]]]=None) -> List[str]:
        bm25_results = self.bm25.retrieve(query, k=2)
        vector_results = self.vector.retrieve(query)[:1]

        def extract_text(result: Any) -> str:
            if isinstance(result, dict):
                return result['text']
            return result
        bm25_texts = [extract_text(r) for r in bm25_results]
        vector_texts = [extract_text(r) for r in vector_results]
        combined = list(dict.fromkeys(bm25_texts + vector_texts))
        if self.mitre_vector and self._is_mitre_query(query) and embed_fn:
            query_embedding = embed_fn([query])[0]
            mitre_results = self.mitre_vector.search(query_embedding, k=4)
            mitre_texts = [hit['text'] for hit in mitre_results]
            for mitre_chunk in mitre_texts:
                if mitre_chunk not in combined:
                    combined.append(mitre_chunk)
        return combined[:4]