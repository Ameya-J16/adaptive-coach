import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS

from rag.embeddings import get_embeddings

load_dotenv()

INDEX_PATH = Path(__file__).parent / "faiss_index"
_vectorstore: FAISS | None = None


def _get_vectorstore() -> FAISS:
    global _vectorstore
    if _vectorstore is None:
        _vectorstore = FAISS.load_local(
            str(INDEX_PATH),
            get_embeddings(),
            allow_dangerous_deserialization=True,
        )
    return _vectorstore


def retrieve(query: str, k: int = 4) -> str:
    """Similarity search with MMR reranking. Returns formatted string of chunks,
    each labelled with the paper it came from so claims stay attributable."""
    vs = _get_vectorstore()
    docs = vs.max_marginal_relevance_search(query, k=k, fetch_k=k * 3)
    parts = []
    for i, doc in enumerate(docs, 1):
        meta = doc.metadata
        cite = meta.get("short_cite", "unattributed")
        doi = meta.get("doi", "")
        label = f"[Source {i} — {cite}"
        if doi:
            label += f", doi:{doi}"
        label += "]"
        parts.append(f"{label}\n{doc.page_content.strip()}")
    return "\n\n---\n\n".join(parts)


def retrieve_docs(query: str, k: int = 4):
    """Return raw Document objects for testing."""
    vs = _get_vectorstore()
    return vs.max_marginal_relevance_search(query, k=k, fetch_k=k * 3)
