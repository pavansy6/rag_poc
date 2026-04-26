class HybridRetriever:
    """
    Combines BM25 (lexical), vector (semantic), and optionally MITRE ATT&CK
    retrieval into a single deduplicated context list.

    MITRE results are pulled only when the query looks threat/attack related,
    so general compliance questions don't get polluted with ATT&CK chunks.
    """

    # keywords that signal the user is asking about MITRE / attack techniques
    MITRE_SIGNALS = [
        "mitre", "att&ck", "attack", "technique", "tactic", "threat actor",
        "adversar", "exploit", "malware", "ransomware", "phishing", "lateral",
        "persistence", "exfiltration", "command and control", "c2", "ttps",
        "initial access", "privilege escalation", "defense evasion",
        "credential", "discovery", "collection", "impact",
        "credential dump", "lsass", "ntds", "sam database",
        "pass the hash", "kerberoast", "mimikatz", "golden ticket",
    ]

    def __init__(self, bm25, vector, mitre_vector=None):
        """
        Args:
            bm25 (BM25Retriever):   lexical retriever (unchanged from your existing code)
            vector (Retriever):     semantic retriever over your docs/ folder
            mitre_vector (FAISSStore | None):
                                    the FAISS store built from MITRE JSON.
                                    Pass None to disable MITRE retrieval.
        """
        self.bm25         = bm25
        self.vector       = vector
        self.mitre_vector = mitre_vector

    def _is_mitre_query(self, query: str) -> bool:
        q = query.lower()
        return any(signal in q for signal in self.MITRE_SIGNALS)

    def retrieve(self, query, embed_fn=None):
        """
        Args:
            query (str):        the user's question
            embed_fn (callable | None):
                                function that takes list[str] → list[list[float]].
                                Required only when mitre_vector is set.

        Returns:
            list[str]: deduplicated context chunks, max 4 total.
        """
        # ── existing sources (unchanged behaviour) ──────────────────────
        bm25_results   = self.bm25.retrieve(query, k=2)
        vector_results = self.vector.retrieve(query)[:1]

        # normalize to plain strings — vector_results may be dicts if
        # FAISSStore.search() was called directly instead of search_texts()
        def to_text(r):
            return r["text"] if isinstance(r, dict) else r

        bm25_results   = [to_text(r) for r in bm25_results]
        vector_results = [to_text(r) for r in vector_results]

        combined = list(dict.fromkeys(bm25_results + vector_results))

        # ── MITRE source (only when relevant + available) ────────────────
        if self.mitre_vector and self._is_mitre_query(query) and embed_fn:
            query_emb    = embed_fn([query])[0]
            mitre_hits   = self.mitre_vector.search(query_emb, k=4)
            mitre_texts  = [h["text"] for h in mitre_hits]

            # add MITRE chunks that aren't already in combined
            for chunk in mitre_texts:
                if chunk not in combined:
                    combined.append(chunk)

        return combined[:4]   # cap total context at 4 chunks