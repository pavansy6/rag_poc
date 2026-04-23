from config import SYSTEM_PROMPT

class RAGEngine:
    def __init__(self, retriever, llm):
        self.retriever = retriever
        self.llm = llm

    def ask(self, query):
        context = self.retriever.retrieve(query)

        prompt = f"""
                Use the context below to answer the question.

                Context:
                {' '.join(context)}

                Question:
                {query}
                """
        return self.llm.generate(prompt)