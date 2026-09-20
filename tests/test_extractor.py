"""Unit and integration tests for document extractor and ingestion pipeline."""
import json
import pytest
from pathlib import Path

from src.config import RAG_DOCS_DIR, EXTRACTED_DATA_DIR
from src.ingestion.extractor import HybridDocumentExtractor
from src.ingestion.models import ElementType, ExtractedDocument
from src.ingestion.pipeline import ExtractionPipeline


@pytest.fixture
def extractor() -> HybridDocumentExtractor:
    return HybridDocumentExtractor()


def test_extractor_initialization(extractor: HybridDocumentExtractor):
    """Test extractor initializes properly with default configuration."""
    assert extractor.config is not None
    assert extractor.config.rag_docs_dir == RAG_DOCS_DIR
    assert len(extractor._header_footer_regexes) > 0


def test_header_footer_detection(extractor: HybridDocumentExtractor):
    """Test running header and footer pattern filtering."""
    assert extractor.is_header_or_footer("Page 1 of 5") is True
    assert extractor.is_header_or_footer("Page 10") is True
    assert extractor.is_header_or_footer("SENIOR AI ENGINEER TECHNICAL ASSESSMENT — SAMPLE CORPUS") is True
    assert extractor.is_header_or_footer("   ") is True
    assert extractor.is_header_or_footer("1. Introduction to RAG") is False
    assert extractor.is_header_or_footer("Vector databases store high-dimensional embeddings.") is False


def test_code_font_classification(extractor: HybridDocumentExtractor):
    """Test classification of monospace fonts."""
    assert extractor.is_code_span("DejaVu-Sans-Mono") is True
    assert extractor.is_code_span("Noto-Sans-Mono") is True
    assert extractor.is_code_span("CourierNew") is True
    assert extractor.is_code_span("Consolas") is True
    assert extractor.is_code_span("Helvetica") is False
    assert extractor.is_code_span("Noto-Sans-Bold") is False


def test_extract_agentic_frameworks_document(extractor: HybridDocumentExtractor):
    """Test extraction of agentic_ai_frameworks.pdf."""
    pdf_path = RAG_DOCS_DIR / "agentic_ai_frameworks.pdf"
    if not pdf_path.is_file():
        pytest.skip(f"Corpus file not found: {pdf_path}")

    doc = extractor.extract_document(pdf_path)
    assert isinstance(doc, ExtractedDocument)
    assert doc.total_pages == 7
    assert len(doc.elements) > 50

    # Verify presence of code elements
    code_elems = [e for e in doc.elements if e.element_type == ElementType.CODE]
    assert len(code_elems) > 0
    # Check that python tool schema or functions are present
    code_contents = "\n".join(e.content for e in code_elems)
    assert "TOOL_SCHEMA" in code_contents or "execute_tool_safely" in code_contents or "rag_chain" in code_contents

    # Verify presence of table elements
    table_elems = [e for e in doc.elements if e.element_type == ElementType.TABLE]
    assert len(table_elems) > 0
    table_contents = "\n".join(e.content for e in table_elems)
    assert "LangChain" in table_contents
    assert "|" in table_contents


def test_extract_vector_database_document(extractor: HybridDocumentExtractor):
    """Test extraction of vector_database_comparison.pdf."""
    pdf_path = RAG_DOCS_DIR / "vector_database_comparison.pdf"
    if not pdf_path.is_file():
        pytest.skip(f"Corpus file not found: {pdf_path}")

    doc = extractor.extract_document(pdf_path)
    assert doc.total_pages == 7
    assert len(doc.elements) > 50

    # Verify tables (benchmarks / comparison matrix)
    table_elems = [e for e in doc.elements if e.element_type == ElementType.TABLE]
    assert len(table_elems) > 0
    table_text = "\n".join(e.content for e in table_elems)
    assert "FAISS" in table_text or "Pinecone" in table_text


def test_extract_rag_architecture_document(extractor: HybridDocumentExtractor):
    """Test extraction of rag_architecture_patterns.pdf."""
    pdf_path = RAG_DOCS_DIR / "rag_architecture_patterns.pdf"
    if not pdf_path.is_file():
        pytest.skip(f"Corpus file not found: {pdf_path}")

    doc = extractor.extract_document(pdf_path)
    assert doc.total_pages == 10
    assert len(doc.elements) > 50

    # Verify headings and sections
    headings = [e for e in doc.elements if e.element_type == ElementType.HEADING]
    assert len(headings) > 5
    heading_text = "\n".join(e.content for e in headings)
    assert "Chunking" in heading_text or "Embedding" in heading_text or "Retrieval" in heading_text


def test_pipeline_execution_and_caching(tmp_path: Path):
    """Test the full pipeline run, artifact writing, and cache mechanism."""
    pipeline = ExtractionPipeline()
    manifest = pipeline.run(
        input_dir=RAG_DOCS_DIR,
        output_dir=tmp_path,
        force_reextract=True,
    )

    assert manifest["status"] == "success"
    assert manifest["total_documents"] == 3
    assert manifest["total_pages"] == 24

    manifest_file = tmp_path / "manifest.json"
    assert manifest_file.is_file()

    # Second run without force should hit cache
    cached_manifest = pipeline.run(
        input_dir=RAG_DOCS_DIR,
        output_dir=tmp_path,
        force_reextract=False,
    )
    assert cached_manifest["total_documents"] == 3
    for d in cached_manifest["documents"]:
        assert d["status"] == "cached"
