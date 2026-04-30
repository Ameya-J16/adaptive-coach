"""
Run with: python -m rag.ingest
Chunks sports_science.md, embeds with OpenAI, and saves a FAISS index.
"""
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

KB_PATH = Path(__file__).parent / "knowledge_base" / "sports_science.md"
INDEX_PATH = Path(__file__).parent / "faiss_index"


def ingest() -> None:
    print(f"Loading knowledge base from {KB_PATH}")
    loader = TextLoader(str(KB_PATH), encoding="utf-8")
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(documents)
    print(f"Split into {len(chunks)} chunks")

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = FAISS.from_documents(chunks, embeddings)

    INDEX_PATH.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(INDEX_PATH))
    print(f"FAISS index saved to {INDEX_PATH}")


if __name__ == "__main__":
    ingest()
