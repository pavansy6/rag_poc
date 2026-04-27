from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage


class Router:
    def __init__(self):
        self.client = ChatOllama(model="qwen2.5:1.5b", temperature=0)

    def route(self, query: str) -> dict:
        prompt = (
            "Classify the following query into one category: cybersecurity, data_science, general.\n"
            f"Query: {query}\nReturn only one word."
        )

        response = self.client.invoke([HumanMessage(content=prompt)])
        category = response.content.strip().lower()

        mapping = {
            "cybersecurity": {"domain": "cyber", "model": "FenkoHQ/Foundation-Sec-8B:latest", "prompt_template": "cyber"},
            "data_science": {"domain": "data_science", "model": "llama3.1:8b", "prompt_template": "ds"},
            "general": {"domain": "general", "model": "qwen2.5:1.5b", "prompt_template": "general"},
        }

        return mapping.get(category, mapping["general"])