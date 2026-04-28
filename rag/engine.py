"""RAG (Retrieval-Augmented Generation) engine that coordinates retrieval and LLM generation.

This module provides :class:`RAGEngine` which ties together a retriever, an
LLM, and a router to produce context-aware responses. Depending on the
classification from the router, the engine may adapt the retrieval query and
select a different prompt template or model for generation.
"""

from typing import Callable, Optional, List
from config import SYSTEM_PROMPT, get_prompt_template


class RAGEngine:
    """Orchestrates retrieval and LLM generation to answer user queries.

    The engine uses a `retriever` to fetch context chunks, formats a prompt
    using a template selected via the `router`, and calls the `llm` to
    generate a final answer.

    Args:
        retriever: Object with a `retrieve(query, embed_fn=...)` method that
            returns an iterable of context strings.
        llm: Object exposing a `generate(prompt, model_name=..., system_prompt=...)`
            method for producing model outputs.
        router: Router-like object with a `route(query)` method returning routing
            metadata (domain, model, prompt_template).
        embed_fn (Optional[Callable]): Optional embedding function passed to the
            retriever.
    """

    def __init__(self, retriever, llm, router, embed_fn: Optional[Callable] = None):
        self.retriever = retriever
        self.llm = llm
        self.router = router
        self.embed_fn = embed_fn

    def ask(self, query: str, return_route: bool = False):
        """Answer a query by retrieving context and invoking the LLM.

        The method first asks the `router` for routing information which can
        alter the retrieval query (for example, prepending "MITRE ATT&CK"
        for cybersecurity queries). It then retrieves context chunks, selects
        an appropriate prompt template, formats it, and calls the LLM to
        generate an answer.

        Args:
            query (str): The user-provided question or instruction.
            return_route (bool): If True, returns a tuple ``(answer, route_info)``.

        Returns:
            str or (str, dict): The generated answer, optionally with routing info.
        """
        route_info = self.router.route(query)

        retrieval_query = query
        # For cybersecurity domain, bias retrieval towards MITRE ATT&CK content.
        if route_info.get("domain") == "cyber":
            retrieval_query = f"MITRE ATT&CK {query}"

        context_chunks = self.retriever.retrieve(retrieval_query, embed_fn=self.embed_fn)
        context_text = "\n\n".join(context_chunks)

        # Choose a prompt template per the router's prompt_template key and model
        prompt_key = route_info.get("prompt_template")
        template = get_prompt_template(prompt_key, route_info.get("model", ""))

        formatted_prompt = template.format(context=context_text, query=query)

        answer = self.llm.generate(formatted_prompt, model_name=route_info["model"], system_prompt=template)

        if return_route:
            return answer, route_info
        return answer