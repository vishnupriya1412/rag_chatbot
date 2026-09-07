"""
rag_chatbot.py — Command-line frontend for the local RAG chatbot.

Run with:  python rag_chatbot.py
"""

from rag_core import build_chatbot


def main():
    print("Loading models and indexing documents (first run downloads ~330MB)...")
    try:
        bot = build_chatbot(docs_folder="sample_docs")
    except FileNotFoundError as e:
        print(str(e))
        return

    print("\nRAG chatbot ready (fully local). Ask questions about your documents (type 'quit' to exit).\n")
    while True:
        question = input("You: ").strip()
        if question.lower() in ("quit", "exit"):
            break
        if not question:
            continue
        result = bot.ask(question)
        print(f"  (retrieved from: {', '.join(result['sources'])})")
        print(f"Bot: {result['answer']}\n")


if __name__ == "__main__":
    main()
