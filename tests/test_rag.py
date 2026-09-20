"""Unit tests for the Simple RAG pipeline components."""
import pytest
from src.ingestion.models import Chunk
from src.ingestion.vector_store import SearchResult
from src.rag.simple_rag import SimpleRAGPipeline, SourceCitation, VerificationResult


def test_source_citation_label():
    """Verify source citation formatting."""
    src = SourceCitation(
        doc_name="vector_database_comparison.pdf",
        page_number=3,
        section="3. Benchmark Results",
        chunk_id="chunk_01",
        score=0.035,
        snippet="FAISS achieved 1.2ms p50 latency.",
    )
    assert src.to_citation_label() == "[vector_database_comparison.pdf, p.3 - 3. Benchmark Results]"


def test_verification_result_grounded():
    """Verify groundness contract."""
    res = VerificationResult(
        status="supported",
        is_grounded=True,
        confidence=0.95,
        supported_claims=["FAISS is an in-memory library"],
        unsupported_claims=[],
        verdict_rationale="All claims backed by retrieved text.",
    )
    assert res.is_grounded is True
    assert res.confidence == 0.95


def test_rerank_and_validate():
    """Test filtering of low-relevance chunks."""
    chunk1 = Chunk(
        chunk_id="c1",
        parent_id="p1",
        doc_name="test.pdf",
        page_number=1,
        section_path="Section A",
        element_type="prose",
        text="Sample high relevance text",
        raw_content="Sample high relevance text",
        token_count=10,
        char_count=26,
    )
    chunk2 = Chunk(
        chunk_id="c2",
        parent_id="p2",
        doc_name="test.pdf",
        page_number=2,
        section_path="Section B",
        element_type="prose",
        text="Low relevance text",
        raw_content="Low relevance text",
        token_count=5,
        char_count=18,
    )
    sr1 = SearchResult(chunk=chunk1, score=0.025)
    sr2 = SearchResult(chunk=chunk2, score=0.005)  # Below min threshold 0.015

    class MockStore:
        pass

    class MockLLM:
        pass

    pipeline = SimpleRAGPipeline(vector_store=MockStore(), llm_client=MockLLM())
    valid_chunks, sources = pipeline._rerank_and_validate([sr1, sr2])

    assert len(valid_chunks) == 1
    assert len(sources) == 1
    assert sources[0].chunk_id == "c1"
