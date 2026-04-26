from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage
from config import SYSTEM_PROMPT

class LLM:
    """
    A wrapper class for interacting with a localized Large Language Model (LLM) using Ollama.
    """
    def __init__(self):
        """
        Initializes the ChatOllama interface.
        Select the appropriate model name matching your local Ollama setup.
        """
        self.llm = ChatOllama(model="qwen2.5:1.5b", temperature=0.0, repetition_penalty=1.3)
        # self.llm = ChatOllama(model="llama3.1:8b", temperature=0.0)
        # self.llm = ChatOllama(model="FenkoHQ/Foundation-Sec-8B:latest")

    def generate(self, prompt):
        """
        Generates a text response based on the SYSTEM_PROMPT and the user query/context.
        
        Args:
            prompt (str): The specific question and retrieved context formatted for the model.
        
        Returns:
            str: The text content of the language model's generated response.
        """
        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=prompt)
        ]
        return self.llm.invoke(messages).content