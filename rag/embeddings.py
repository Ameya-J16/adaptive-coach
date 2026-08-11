"""Embedding backend for the RAG index.

Default is `local`: BAAI/bge-base-en-v1.5 running in-process via fastembed
(ONNX, no PyTorch). That removes the last paid API call from the retrieval
path — embeddings cost nothing and work offline — and keeps the memory
footprint small enough for free hosting tiers.

    EMBEDDING_PROVIDER=local     # local | openai
    EMBEDDING_MODEL=BAAI/bge-base-en-v1.5

Measured trade-off (14-case critic eval, GPT-4o held constant):

    openai  text-embedding-3-small   accuracy 1.00   flaw detection 1.00
    local   bge-base-en-v1.5         accuracy 0.90   flaw detection 0.75
    local   bge-small-en-v1.5        accuracy 0.80   flaw detection 0.75

False-pass rate stayed 0.00 in all three: the deterministic checks in
graph/plan_checks.py catch mechanical violations regardless of retrieval
quality, so a weaker backend costs recall on soft judgements, never safety.

Embeddings are a rounding error on cost (~$0.005 per full re-ingest, ~$1e-6
per plan) — the LLM is ~100% of the bill. So `openai` here is a reasonable
choice even on a budget; `local` is for running with no paid API at all.

The index is embedding-specific: vectors from different models are not
comparable, so changing this requires re-running `python -m rag.ingest`.
"""
import os

from dotenv import load_dotenv

load_dotenv()

DEFAULT_LOCAL_MODEL = "BAAI/bge-base-en-v1.5"
DEFAULT_OPENAI_MODEL = "text-embedding-3-small"


def active_provider() -> str:
    return os.getenv("EMBEDDING_PROVIDER", "local").strip().lower()


def embedding_model_name() -> str:
    provider = active_provider()
    default = DEFAULT_LOCAL_MODEL if provider == "local" else DEFAULT_OPENAI_MODEL
    return os.getenv("EMBEDDING_MODEL", default)


def get_embeddings():
    """Return a LangChain embeddings object for the configured provider."""
    provider = active_provider()
    model = embedding_model_name()

    if provider == "openai":
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings(model=model)

    if provider == "local":
        from langchain_community.embeddings import FastEmbedEmbeddings
        # Downloads the ONNX model once (~130MB) into the fastembed cache.
        return FastEmbedEmbeddings(model_name=model)

    raise ValueError(
        f"Unknown EMBEDDING_PROVIDER {provider!r}. Use 'local' or 'openai'."
    )


def describe() -> str:
    provider = active_provider()
    cost = "free, in-process" if provider == "local" else "paid API"
    return f"{provider} · {embedding_model_name()} · {cost}"
