from config import SYSTEM_PROMPT

class RAGEngine:
    def __init__(self, retriever, llm):
        self.retriever = retriever
        self.llm = llm

    def ask(self, query):
        context = self.retriever.retrieve(query)

        prompt = f"""
                Answer ONLY using the provided context.

                If the answer is not explicitly available in the context, say:
                'The information is not available in the provided documents.'

                Context:
                {' '.join(context)}

                Question:
                {query}
                """

        return self.llm.generate(prompt)