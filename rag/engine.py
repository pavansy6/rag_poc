"""RAG (Retrieval-Augmented Generation) engine that coordinates retrieval and LLM generation.

This module provides :class:`RAGEngine` which retrieves context and generates
answers using either the cybersecurity model (if relevant documents found) or
a fallback model (if no relevant context).
"""

from typing import Callable, Optional, List
from config import SYSTEM_PROMPT, get_prompt_template


class RAGEngine:
    """Orchestrates retrieval and LLM generation to answer user queries.

    The engine retrieves context chunks first. If relevant content is found,
    it uses the cybersecurity model with cyber prompt. If no relevant content
    is found, it falls back to llama3.1:8b for general knowledge.

    Args:
        retriever: Object with a `retrieve(query, embed_fn=...)` method that
            returns an iterable of context strings.
        llm: Object exposing a `generate(prompt, model_name=..., system_prompt=...)`
            method for producing model outputs.
        embed_fn (Optional[Callable]): Optional embedding function passed to the
            retriever.
    """

    def __init__(self, retriever, llm, embed_fn: Optional[Callable] = None):
        self.retriever = retriever
        self.llm = llm
        self.embed_fn = embed_fn

    def ask(self, query: str, return_route: bool = False, conversation_history: Optional[List[dict]] = None):
        """Answer a query by retrieving context first, then deciding on model/prompt.

        The method retrieves context chunks first. If relevant content is found,
        it uses the cybersecurity model (foundation-sec:8b) with cyber prompt.
        If no relevant content is found, it falls back to llama3.1:8b for general
        knowledge. Conversation history is included to maintain multi-turn context.

        Args:
            query (str): The user-provided question or instruction.
            return_route (bool): If True, returns a tuple ``(answer, route_info)``.
            conversation_history (Optional[List[dict]]): List of previous messages in the
                format [{"role": "user"|"assistant", "content": str}, ...]. Used to
                provide context for multi-turn conversations.

        Returns:
            str or (str, dict): The generated answer, optionally with routing info.
        """
        # Retrieve context first
        context_chunks = self.retriever.retrieve(query, embed_fn=self.embed_fn)
        context_text = "\n\n".join(context_chunks)

        # Decide on model and prompt based on retrieval results
        if context_chunks:
            # Found relevant content - use cyber model with cyber prompt
            model = "qwen2.5:1.5b"
            prompt_key = "cyber"
            domain = "cyber"
        else:
            # No relevant content - fallback to llama3.1:8b
            model = "llama3.1:8b"
            prompt_key = "general"
            domain = "fallback"

        template = get_prompt_template(prompt_key, model)

        # Format prompt with context, conversation history, and query
        formatted_prompt = self._format_prompt_with_history(
            template, context=context_text, query=query, history=conversation_history
        )

        answer = self.llm.generate(formatted_prompt, model_name=model, system_prompt=template)

        route_info = {"domain": domain, "model": model, "prompt_template": prompt_key}

        if return_route:
            return answer, route_info
        return answer

    def _format_prompt_with_history(
        self, template: str, context: str, query: str, history: Optional[List[dict]] = None
    ) -> str:
        """Format the prompt template with context, conversation history, and query.

        If conversation history is provided, it is inserted between the CONTEXT
        and QUESTION sections to give the LLM awareness of previous exchanges.

        Args:
            template (str): The prompt template (with {context}, {query}, and optionally {history}).
            context (str): The retrieved context text.
            query (str): The current user query.
            history (Optional[List[dict]]): Conversation history as list of {"role": ..., "content": ...}.

        Returns:
            str: The formatted prompt ready for the LLM.
        """
        # Build conversation history string if provided
        history_text = ""
        if history and len(history) > 0:
            # Include last 4 exchanges (8 messages) to avoid token bloat
            recent_history = history[-8:]
            history_lines = []
            for msg in recent_history:
                role = msg.get("role", "user").upper()
                content = msg.get("content", "").strip()
                history_lines.append(f"{role}: {content}")
            history_text = "\n".join(history_lines)

        # Try to format with history if template supports it, otherwise fall back to basic format
        try:
            formatted = template.format(context=context, query=query, history=history_text)
        except KeyError:
            # Template doesn't have {history} placeholder, use standard format
            formatted = template.format(context=context, query=query)

        return formatted