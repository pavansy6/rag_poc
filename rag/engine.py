from typing import Callable, Optional, List
from config import SYSTEM_PROMPT, get_prompt_template


class RAGEngine:
    def __init__(self, retriever, llm, router, embed_fn: Optional[Callable] = None):
        self.retriever = retriever
        self.llm = llm
        self.router = router
        self.embed_fn = embed_fn

    def ask(self, query: str, return_route: bool = False):
        route_info = self.router.route(query)

        retrieval_query = query
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