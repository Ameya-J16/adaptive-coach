"""Tests for RAG retrieval — requires a built FAISS index (run python -m rag.ingest first)."""
import sys
import os
import pytest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

INDEX_PATH = os.path.join(os.path.dirname(__file__), "..", "rag", "faiss_index")


def _index_exists() -> bool:
    return os.path.isdir(INDEX_PATH) and any(os.scandir(INDEX_PATH))


@pytest.mark.skipif(not _index_exists(), reason="FAISS index not built — run python -m rag.ingest first")
class TestRAGRetriever:
    def test_retriever_returns_non_empty_for_deload(self):
        """Querying 'deload protocol' should return non-empty results."""
        from rag.retriever import retrieve
        result = retrieve("deload protocol")
        assert result, "Expected non-empty string from retrieve()"
        assert len(result) > 50, "Expected meaningful content, got too short response"

    def test_retriever_returns_k_distinct_chunks(self):
        """MMR search should return k distinct chunks (no duplicates)."""
        from rag.retriever import retrieve_docs
        docs = retrieve_docs("progressive overload volume increase", k=4)
        assert len(docs) >= 2, f"Expected at least 2 chunks, got {len(docs)}"

        # Check that chunks are distinct (MMR should prevent duplicates)
        contents = [d.page_content for d in docs]
        unique_contents = set(contents)
        assert len(unique_contents) == len(contents), "MMR search returned duplicate chunks"

    def test_retriever_relevant_to_acwr(self):
        """Querying ACWR should return content mentioning workload ratio."""
        from rag.retriever import retrieve
        result = retrieve("acute chronic workload ratio injury risk")
        assert "acwr" in result.lower() or "workload" in result.lower() or "chronic" in result.lower(), \
            "Expected ACWR-related content in retrieval results"

    def test_retriever_nutrition_query(self):
        """Nutrition-related queries should return protein/carb content."""
        from rag.retriever import retrieve
        result = retrieve("protein requirements athletes carbohydrates")
        assert any(kw in result.lower() for kw in ["protein", "carb", "calorie", "nutrition"]), \
            "Expected nutrition-related content in retrieval results"

    def test_retrieve_returns_source_markers(self):
        """Retrieved content should include source markers for transparency."""
        from rag.retriever import retrieve
        result = retrieve("recovery sleep training")
        assert "[Source" in result, "Expected source markers in retrieval output"
