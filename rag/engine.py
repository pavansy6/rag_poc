from typing import List, Callable, Optional
from config import SYSTEM_PROMPT

class RAGEngine:
    MITRE_SIGNALS = ['mitre', 'att&ck', 'attack', 'technique', 'tactic', 'threat actor', 'adversar', 'exploit', 'malware', 'ransomware', 'phishing', 'lateral', 'persistence', 'exfiltration', 'command and control', 'c2', 'ttps', 'initial access', 'privilege escalation', 'defense evasion', 'credential', 'discovery', 'collection', 'impact', 'credential dump', 'lsass', 'ntds', 'sam database', 'pass the hash', 'kerberoast', 'mimikatz']

    def __init__(self, retriever, llm, embed_fn: Optional[Callable[[List[str]], List[List[float]]]]=None):
        self.retriever = retriever
        self.llm = llm
        self.embed_fn = embed_fn

    def _is_mitre_like(self, query: str) -> bool:
        query_lower = query.lower()
        return any((signal in query_lower for signal in self.MITRE_SIGNALS))

    def ask(self, query: str) -> str:
        retrieval_query = f'MITRE ATT&CK {query}' if self._is_mitre_like(query) else query
        context_chunks = self.retriever.retrieve(retrieval_query, embed_fn=self.embed_fn)
        context_text = '\n\n'.join(context_chunks)
        prompt = SYSTEM_PROMPT.format(context=context_text, query=query)
        return self.llm.generate(prompt)