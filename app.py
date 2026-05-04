"""Build and run the RAG pipeline for document and MITRE retrieval.

This module provides a simple interactive command-line interface for querying
the RAG pipeline. The core pipeline building logic has been consolidated into
rag/pipeline_builder.py to avoid duplication with streamlit_app.py.
"""

from rag.pipeline_builder import build_rag_pipeline

def run_interactive_loop(engine):
    """Run a terminal-based interactive query loop with conversation history.

    Maintains a conversation history across multiple queries to enable
    context-aware responses for follow-up questions.

    Args:
        engine: A RAG engine exposing an ``ask`` method for generating answers.
    """
    conversation_history = []
    
    while True:
        try:
            query = input('Ask: ').strip()
            if query.lower() in ('exit', 'quit'):
                break
            if not query:
                continue
            
            # Pass conversation history to the RAG engine for context awareness
            answer = engine.ask(query, conversation_history=conversation_history)
            print(f'\n{answer}\n')
            
            # Append this exchange to the conversation history
            conversation_history.append({"role": "user", "content": query})
            conversation_history.append({"role": "assistant", "content": answer})
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f'Error: {e}')


if __name__ == '__main__':
    engine = build_rag_pipeline()
    run_interactive_loop(engine)