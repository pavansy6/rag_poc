from config import SYSTEM_PROMPT


class RAGEngine:
    """
    The core Retrieval-Augmented Generation engine. Retrieves context chunks
    and builds a prompt for the LLM.
    """

    # reuse the same signal list as HybridRetriever — no duplication
    MITRE_SIGNALS = [
        "mitre", "att&ck", "attack", "technique", "tactic", "threat actor",
        "adversar", "exploit", "malware", "ransomware", "phishing", "lateral",
        "persistence", "exfiltration", "command and control", "c2", "ttps",
        "initial access", "privilege escalation", "defense evasion",
        "credential", "discovery", "collection", "impact",
        "credential dump", "lsass", "ntds", "sam database",
        "pass the hash", "kerberoast", "mimikatz",
    ]

    def __init__(self, retriever, llm, embed_fn=None):
        self.retriever = retriever
        self.llm       = llm
        self.embed_fn  = embed_fn

    def _is_mitre_like(self, query: str) -> bool:
        q = query.lower()
        return any(signal in q for signal in self.MITRE_SIGNALS)

    def ask(self, query):
        # prepend MITRE context to the query embedding so tactic/technique
        # chunks rank higher for attack-related questions
        expanded = f"MITRE ATT&CK {query}" if self._is_mitre_like(query) else query

        context_chunks = self.retriever.retrieve(expanded, embed_fn=self.embed_fn)
        context_text   = "\n\n".join(context_chunks)

        # use original query in the prompt — only expansion affects retrieval
        prompt = SYSTEM_PROMPT.format(context=context_text, query=query)

        return self.llm.generate(prompt)