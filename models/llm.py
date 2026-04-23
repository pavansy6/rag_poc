from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage
from config import SYSTEM_PROMPT

class LLM:
    def __init__(self):
        self.llm = ChatOllama(model="qwen2.5:1.5b")

    def generate(self, prompt):
        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=prompt)
        ]
        return self.llm.invoke(messages).content