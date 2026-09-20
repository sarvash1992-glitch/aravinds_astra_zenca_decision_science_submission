"""Unit and integration tests for HybridSemanticStructuralChunker."""
import json
import pytest
from pathlib import Path

from src.config import EXTRACTED_DATA_DIR, CHUNKS_DATA_DIR
from src.ingestion.chunker import HybridSemanticStructuralChunker
from src.ingestion.models import Chunk, ElementType, ExtractedDocument


@pytest.fixture
def chunker() -> HybridSemanticStructuralChunker:
    return HybridSemanticStructuralChunker()


def test_chunker_initialization(chunker: HybridSemanticStructuralChunker):
    """Test chunker initializes and counts tokens properly."""
    assert chunker.config is not None
    tokens = chunker.count_tokens("Hello world! This is a test.")
    assert tokens > 0
    assert chunker.count_tokens("") == 0


def test_breadcrumb_injection(chunker: HybridSemanticStructuralChunker):
    """Test contextual breadcrumb creation."""
    bc = chunker._format_breadcrumb("vector_database_comparison.pdf", "8. Performance Benchmarks", 5)
    assert "[Document: Vector Database Comparison" in bc
    assert "Section: 8. Performance Benchmarks" in bc
    assert "Page: 5" in bc


def test_chunk_document_agentic_frameworks(chunker: HybridSemanticStructuralChunker):
    """Test chunking agentic_ai_frameworks.json."""
    json_path = EXTRACTED_DATA_DIR / "agentic_ai_frameworks.json"
    if not json_path.is_file():
        pytest.skip(f"Extracted file not found: {json_path}")

    with open(json_path, "r", encoding="utf-8") as f:
        doc = ExtractedDocument.model_validate(json.load(f))

    chunks = chunker.chunk_document(doc)
    assert len(chunks) > 20

    # Verify table chunks
    table_chunks = [c for c in chunks if c.element_type == "table"]
    assert len(table_chunks) > 0
    for tc in table_chunks:
        assert "|" in tc.text
        assert "[Document:" in tc.text

    # Verify code chunks
    code_chunks = [c for c in chunks if c.element_type == "code"]
    assert len(code_chunks) > 0
    for cc in code_chunks:
        assert "```" in cc.text

    # Verify token sizes
    for c in chunks:
        assert c.token_count <= 800  # Hard ceiling


def test_chunk_document_vector_db(chunker: HybridSemanticStructuralChunker):
    """Test chunking vector_database_comparison.json."""
    json_path = EXTRACTED_DATA_DIR / "vector_database_comparison.json"
    if not json_path.is_file():
        pytest.skip(f"Extracted file not found: {json_path}")

    with open(json_path, "r", encoding="utf-8") as f:
        doc = ExtractedDocument.model_validate(json.load(f))

    chunks = chunker.chunk_document(doc)
    assert len(chunks) > 20

    # Ensure metadata filters and benchmarks are represented
    all_text = " ".join(c.text for c in chunks)
    assert "FAISS" in all_text
    assert "Pinecone" in all_text
    assert "pgvector" in all_text


def test_chunk_document_rag_patterns(chunker: HybridSemanticStructuralChunker):
    """Test chunking rag_architecture_patterns.json."""
    json_path = EXTRACTED_DATA_DIR / "rag_architecture_patterns.json"
    if not json_path.is_file():
        pytest.skip(f"Extracted file not found: {json_path}")

    with open(json_path, "r", encoding="utf-8") as f:
        doc = ExtractedDocument.model_validate(json.load(f))

    chunks = chunker.chunk_document(doc)
    assert len(chunks) > 20

    all_text = " ".join(c.text for c in chunks)
    assert "Chunking" in all_text or "Retrieval" in all_text
