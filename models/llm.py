from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage
from config import SYSTEM_PROMPT
from typing import Optional


class LLM:
    def generate(self, prompt: str, model_name: str, system_prompt: Optional[str] = None) -> str:
        client = ChatOllama(model=model_name, temperature=0.0, repeat_penalty=1.2)

        # System message: pass the template (unformatted) so model sees framing.
        sys_msg = system_prompt or SYSTEM_PROMPT
        messages = [SystemMessage(content=sys_msg), HumanMessage(content=prompt)]

        response = client.invoke(messages)
        return response.content