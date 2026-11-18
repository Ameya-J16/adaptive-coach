"""Build the FAISS index over the sports-science corpus.

Run with: python -m rag.ingest

Indexes two things:
  1. rag/knowledge_base/corpus/*.md  — real peer-reviewed CC BY papers fetched
     by scripts/build_corpus.py, each carrying citation frontmatter
  2. rag/knowledge_base/sports_science.md — the hand-written practitioner
     summary, kept as a concise index layer over the primary literature

Every chunk is prefixed with its citation so retrieval returns attributable
evidence: the critic can check a claim against a named paper rather than
against unattributed prose.
"""
import re
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag.embeddings import describe as describe_embeddings, get_embeddings

load_dotenv()

KB_DIR = Path(__file__).parent / "knowledge_base"
CORPUS_DIR = KB_DIR / "corpus"
SUMMARY_PATH = KB_DIR / "sports_science.md"
INDEX_PATH = Path(__file__).parent / "faiss_index"

FRONTMATTER = re.compile(r"^---\n(.*?)\n---\n", re.S)


def _parse_frontmatter(text: str) -> tuple[dict, str]:
    m = FRONTMATTER.match(text)
    if not m:
        return {}, text
    meta = {}
    for line in m.group(1).splitlines():
        if ":" not in line:
            continue
        key, _, value = line.partition(":")
        meta[key.strip()] = value.strip().strip('"')
    return meta, text[m.end():]


def load_corpus_documents() -> list[Document]:
    docs = []
    for path in sorted(CORPUS_DIR.glob("*.md")):
        if path.name == "LICENSES.md":
            continue
        meta, body = _parse_frontmatter(path.read_text(encoding="utf-8"))
        if not meta.get("title"):
            continue
        authors = meta.get("authors", "")
        short_author = authors.split(",")[0].strip() if authors else "Unknown"
        docs.append(Document(
            page_content=body,
            metadata={
                "source": path.name,
                "kind": "paper",
                "title": meta.get("title", ""),
                "authors": authors,
                "journal": meta.get("journal", ""),
                "year": meta.get("year", ""),
                "doi": meta.get("doi", ""),
                "pmcid": meta.get("pmcid", ""),
                "license": meta.get("license", ""),
                "topic": meta.get("topic", ""),
                "short_cite": f"{short_author} et al., {meta.get('year', 'n.d.')}",
            },
        ))
    return docs


def load_summary_document() -> list[Document]:
    if not SUMMARY_PATH.exists():
        return []
    return [Document(
        page_content=SUMMARY_PATH.read_text(encoding="utf-8"),
        metadata={
            "source": SUMMARY_PATH.name,
            "kind": "summary",
            "title": "AdaptiveCoach practitioner summary",
            "short_cite": "AdaptiveCoach practitioner summary",
            "license": "project",
        },
    )]


def ingest() -> None:
    corpus_docs = load_corpus_documents()
    summary_docs = load_summary_document()
    print(f"Loaded {len(corpus_docs)} corpus papers and {len(summary_docs)} summary document(s)")
    if not corpus_docs:
        print("WARNING: no corpus papers found — run python -m scripts.build_corpus first")

    # Papers are dense prose: larger chunks keep an argument intact. The summary
    # is terse and factual, so it keeps the original tighter chunking.
    paper_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=150)
    summary_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

    chunks = paper_splitter.split_documents(corpus_docs)
    chunks += summary_splitter.split_documents(summary_docs)

    # Prefix each chunk with its citation so the text the LLM sees is attributable
    # even after the vector store hands back a bare fragment.
    for c in chunks:
        cite = c.metadata.get("short_cite", "unattributed")
        title = c.metadata.get("title", "")
        journal = c.metadata.get("journal", "")
        header = f"[{cite}"
        if journal:
            header += f", {journal}"
        header += f"] {title}".rstrip()
        c.page_content = f"{header}\n{c.page_content.strip()}"

    print(f"Split into {len(chunks)} chunks")

    print(f"Embeddings: {describe_embeddings()}")
    vectorstore = FAISS.from_documents(chunks, get_embeddings())

    INDEX_PATH.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(INDEX_PATH))
    print(f"FAISS index saved to {INDEX_PATH}")


if __name__ == "__main__":
    ingest()
