"""Unit tests for GeminiEmbedder and HybridVectorStore."""
import pytest
import numpy as np

from src.config import embedding_config
from src.ingestion.embedder import GeminiEmbedder
from src.ingestion.models import Chunk
from src.ingestion.vector_store import HybridVectorStore


@pytest.fixture
def embedder() -> GeminiEmbedder:
    return GeminiEmbedder()


def test_embedder_dimension(embedder: GeminiEmbedder):
    """Test embedding single query and document batch."""
    vec = embedder.embed_query("Vector database comparison test")
    assert isinstance(vec, np.ndarray)
    assert vec.shape == (3072,)
    # Verify L2 normalized
    norm = np.linalg.norm(vec)
    assert pytest.approx(norm, abs=1e-3) == 1.0


def test_hybrid_vector_store_in_memory(embedder: GeminiEmbedder, tmp_path):
    """Test building, saving, loading and querying the hybrid store."""
    sample_chunks = [
        Chunk(
            chunk_id="test_c1",
            parent_id="test_s1",
            doc_name="vector_database_comparison.pdf",
            page_number=5,
            section_path="8. Performance Benchmarks",
            element_type="table",
            text="[Document: Vector DB | Section: Benchmarks | Page: 5]\nFAISS HNSW latency is 2-5ms at 1M vectors.",
            raw_content="FAISS HNSW latency is 2-5ms at 1M vectors.",
            token_count=20,
            char_count=45,
        ),
        Chunk(
            chunk_id="test_c2",
            parent_id="test_s2",
            doc_name="agentic_ai_frameworks.pdf",
            page_number=2,
            section_path="3. LangGraph",
            element_type="prose",
            text="[Document: Agentic AI | Section: LangGraph | Page: 2]\nLangGraph replaces LangChain AgentExecutor with an explicit state graph.",
            raw_content="LangGraph replaces LangChain AgentExecutor with an explicit state graph.",
            token_count=22,
            char_count=73,
        ),
    ]

    store = HybridVectorStore(embedder=embedder)
    store.build_index(sample_chunks)
    assert store.faiss_index.ntotal == 2

    # Test saving and loading
    store.save(tmp_path)
    loaded_store = HybridVectorStore(embedder=embedder)
    loaded_store.load(tmp_path)
    assert loaded_store.faiss_index.ntotal == 2
    assert len(loaded_store.chunks) == 2

    # Test Hybrid RRF Search
    results = loaded_store.search_hybrid("FAISS latency benchmarks", top_k=1)
    assert len(results) == 1
    assert results[0].chunk.chunk_id == "test_c1"
    assert results[0].score > 0
