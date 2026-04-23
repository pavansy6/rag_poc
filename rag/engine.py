from config import SYSTEM_PROMPT

class RAGEngine:
    """
    The core Retrieval-Augmented Generation context engine. It intercepts user queries, 
    fetches relevant context via the retriever, and builds a composite prompt for the LLM.
    """
    def __init__(self, retriever, llm):
        """
        Initializes the RAG Engine with a retriever implementation and an LLM instance.
        """
        self.retriever = retriever
        self.llm = llm

    def ask(self, query):
        """
        Processes a user query by retrieving relevant context and generating a response.
        
        Args:
            query (str): The user's specific question.
            
        Returns:
            str: The final generated response from the LLM.
        """
        # Retrieve context from vector store / BM25
        context = self.retriever.retrieve(query)

        # Assemble the prompt with context injection
        prompt = f"""
                Use the context below to answer the question.

                Context:
                {' '.join(context)}

                Question:
                {query}
                """
        # Generate the response
        return self.llm.generate(prompt)