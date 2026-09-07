"""
rag_core.py — Shared RAG logic used by the web frontend (and CLI, if used).

Nothing in this file talks to a terminal or a browser — it's pure logic:
loading/parsing documents, chunking, embedding, retrieval, and generation.
"""

import os
import glob
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from pypdf import PdfReader
import docx


# --------------------------------------------------------------------------
# Text extraction — supports .txt, .pdf, .docx
# --------------------------------------------------------------------------

def extract_text_from_pdf(filepath: str) -> str:
    reader = PdfReader(filepath)
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def extract_text_from_docx(filepath: str) -> str:
    document = docx.Document(filepath)
    return "\n".join(p.text for p in document.paragraphs)


def extract_text_from_txt(filepath: str) -> str:
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def extract_text(filepath: str) -> str:
    """Dispatch to the right parser based on file extension."""
    ext = os.path.splitext(filepath)[1].lower()
    if ext == ".pdf":
        return extract_text_from_pdf(filepath)
    elif ext == ".docx":
        return extract_text_from_docx(filepath)
    elif ext == ".txt":
        return extract_text_from_txt(filepath)
    else:
        raise ValueError(f"Unsupported file type: {ext} (supported: .txt, .pdf, .docx)")


def load_documents(folder_path: str) -> list[dict]:
    """Load all supported files from a folder. Returns list of {source, text}."""
    docs = []
    patterns = ["*.txt", "*.pdf", "*.docx"]
    for pattern in patterns:
        for filepath in glob.glob(os.path.join(folder_path, pattern)):
            try:
                text = extract_text(filepath)
                if text.strip():
                    docs.append({"source": os.path.basename(filepath), "text": text})
            except Exception as e:
                print(f"Skipping {filepath}: {e}")
    return docs


# --------------------------------------------------------------------------
# Chunking
# --------------------------------------------------------------------------

def chunk_text(text: str, chunk_size: int = 200, overlap: int = 30) -> list[str]:
    """Split text into overlapping word chunks so context isn't cut off mid-idea."""
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        if chunk.strip():
            chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


# --------------------------------------------------------------------------
# Vector store (simple, in-memory, supports incremental additions)
# --------------------------------------------------------------------------

class VectorStore:
    def __init__(self, embed_model: SentenceTransformer):
        self.embed_model = embed_model
        self.chunks: list[str] = []
        self.sources: list[str] = []
        self.embeddings: np.ndarray | None = None

    def add_documents(self, docs: list[dict]) -> int:
        """Adds new documents to the store (on top of any existing ones).
        Returns the number of chunks added."""
        new_chunks = []
        new_sources = []
        for doc in docs:
            for chunk in chunk_text(doc["text"]):
                new_chunks.append(chunk)
                new_sources.append(doc["source"])

        if not new_chunks:
            return 0

        new_embeddings = self.embed_model.encode(
            new_chunks, normalize_embeddings=True, show_progress_bar=False
        )

        self.chunks.extend(new_chunks)
        self.sources.extend(new_sources)
        self.embeddings = (
            new_embeddings if self.embeddings is None
            else np.vstack([self.embeddings, new_embeddings])
        )
        return len(new_chunks)

    def list_sources(self) -> list[str]:
        return sorted(set(self.sources))

    def search(self, query: str, top_k: int = 3) -> list[dict]:
        """Return the top_k most relevant chunks for a query."""
        if self.embeddings is None or len(self.chunks) == 0:
            return []
        query_vec = self.embed_model.encode([query], normalize_embeddings=True)[0]
        scores = self.embeddings @ query_vec  # cosine similarity (vectors normalized)
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [
            {"text": self.chunks[i], "source": self.sources[i], "score": float(scores[i])}
            for i in top_indices
        ]


# --------------------------------------------------------------------------
# RAG chatbot: retrieval + local LLM generation
# --------------------------------------------------------------------------

class RAGChatbot:
    def __init__(self, vector_store: VectorStore, generator):
        self.vector_store = vector_store
        self.generator = generator

    def _build_prompt(self, question: str, retrieved_chunks: list[dict]) -> str:
        context = "\n".join(c["text"] for c in retrieved_chunks)
        return (
            "Answer the question using only the context below. "
            "If the answer isn't in the context, say you don't know.\n\n"
            f"Context: {context}\n\n"
            f"Question: {question}"
        )

    def ask(self, question: str, top_k: int = 3) -> dict:
        """Returns {'answer': str, 'sources': list[str]}."""
        retrieved = self.vector_store.search(question, top_k=top_k)
        if not retrieved:
            return {
                "answer": "No documents have been uploaded yet — add one first.",
                "sources": [],
            }

        prompt = self._build_prompt(question, retrieved)
        result = self.generator(prompt, max_new_tokens=200, do_sample=False)
        answer = result[0]["generated_text"].strip()

        sources_used = sorted(set(c["source"] for c in retrieved))
        return {"answer": answer, "sources": sources_used}


# --------------------------------------------------------------------------
# Convenience builder
# --------------------------------------------------------------------------

def build_chatbot(docs_folder: str | None = None) -> RAGChatbot:
    """Loads models and returns a ready-to-use chatbot.
    If docs_folder is given and contains files, they're indexed immediately;
    otherwise the store starts empty and documents can be added later
    via vector_store.add_documents()."""
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    generator = pipeline("text2text-generation", model="google/flan-t5-base")

    store = VectorStore(embed_model)

    if docs_folder:
        docs = load_documents(docs_folder)
        if docs:
            store.add_documents(docs)

    return RAGChatbot(store, generator)
