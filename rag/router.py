"""Routing utilities for selecting an LLM and prompt template by query category.

This module exposes the :class:`Router` which classifies an incoming query
into a simple category and returns routing metadata used to select a model
and prompt template.

The classification is delegated to an Ollama chat model that returns a single
word category. The returned mapping contains the chosen domain, model name,
and a short prompt template identifier.
"""

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage


class Router:
    """Simple query router that classifies text into a domain and returns routing metadata.

    The router sends a short prompt to an Ollama chat model to classify the
    incoming `query` into one of three categories: ``cybersecurity``,
    ``data_science``, or ``general``. It then maps the returned category to a
    dictionary containing the chosen `domain`, `model`, and `prompt_template`.

    Attributes:
        client (ChatOllama): The Ollama chat client used for classification.
    """

    def __init__(self):
        """Create a Router with a default Ollama client.

        The Ollama client is configured with a default model and temperature.
        No parameters are required.
        """
        self.client = ChatOllama(model="qwen2.5:1.5b", temperature=0)

    def route(self, query: str) -> dict:
        """Classify a query and return a routing dictionary.

        Sends a classification prompt to the Ollama model and expects a single
        word category in the response content. The category is mapped to a
        routing dictionary with keys: ``domain``, ``model``, and
        ``prompt_template``.

        Args:
            query (str): The user's free-text query to classify.

        Returns:
            dict: A mapping with routing information for the chosen category.
                If the model returns an unknown category, the ``general``
                mapping is returned as a safe default.

        Raises:
            Any exceptions raised by the underlying Ollama client when
            invoking the model (left uncaught to allow callers to handle
            retries/logging as appropriate).
        """
        prompt = (
            "Classify the following query into one category: cybersecurity, data_science, general.\n"
            f"Query: {query}\nReturn only one word."
        )

        response = self.client.invoke([HumanMessage(content=prompt)])
        category = response.content.strip().lower()

        mapping = {
            "cybersecurity": {"domain": "cyber", "model": "qwen2.5:1.5b", "prompt_template": "cyber"},
            "data_science": {"domain": "data_science", "model": "llama3.1:8b", "prompt_template": "ds"},
            "general": {"domain": "general", "model": "qwen2.5:1.5b", "prompt_template": "general"},
        }

        return mapping.get(category, mapping["general"])