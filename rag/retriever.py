import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()

INDEX_PATH = Path(__file__).parent / "faiss_index"
_vectorstore: FAISS | None = None


def _get_vectorstore() -> FAISS:
    global _vectorstore
    if _vectorstore is None:
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        _vectorstore = FAISS.load_local(
            str(INDEX_PATH),
            embeddings,
            allow_dangerous_deserialization=True,
        )
    return _vectorstore


def retrieve(query: str, k: int = 4) -> str:
    """Similarity search with MMR reranking. Returns formatted string of chunks."""
    vs = _get_vectorstore()
    docs = vs.max_marginal_relevance_search(query, k=k, fetch_k=k * 3)
    parts = []
    for i, doc in enumerate(docs, 1):
        parts.append(f"[Source {i}]\n{doc.page_content.strip()}")
    return "\n\n---\n\n".join(parts)


def retrieve_docs(query: str, k: int = 4):
    """Return raw Document objects for testing."""
    vs = _get_vectorstore()
    return vs.max_marginal_relevance_search(query, k=k, fetch_k=k * 3)
