"""Data models for extracted document components, chunks, and metadata."""
from enum import Enum
from typing import Any
from datetime import datetime, timezone
from pydantic import BaseModel, Field


class ElementType(str, Enum):
    """Categorization of document content elements."""
    HEADING = "heading"
    PARAGRAPH = "paragraph"
    TABLE = "table"
    CODE = "code"
    LIST_ITEM = "list_item"


class DocumentElement(BaseModel):
    """An individual structured element extracted from a document page."""
    id: str = Field(description="Unique element identifier, e.g. doc_p1_e3")
    page_number: int = Field(description="1-indexed page number where element appears")
    element_type: ElementType = Field(description="Type of the element (heading, paragraph, table, code)")
    content: str = Field(description="Normalized text or markdown content of the element")
    section: str = Field(default="", description="Hierarchical section path, e.g. '1. Introduction > 1.1 What is RAG'")
    bbox: tuple[float, float, float, float] | None = Field(default=None, description="(x0, y0, x1, y1) bounding box on the page")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Supplementary metadata (e.g. font, language, table dims)")


class ExtractedTable(BaseModel):
    """Structured representation of a detected table."""
    headers: list[str] = Field(default_factory=list, description="Column header titles")
    rows: list[list[str]] = Field(default_factory=list, description="Table cell rows")
    markdown: str = Field(description="Formatted GitHub-flavored Markdown representation")
    bbox: tuple[float, float, float, float] | None = Field(default=None)


class ExtractedDocument(BaseModel):
    """Complete extracted document payload preserving structure and metadata."""
    doc_name: str = Field(description="Base filename of the document")
    source_path: str = Field(description="Absolute path to the source document")
    total_pages: int = Field(description="Number of pages in the source document")
    file_hash: str = Field(description="SHA256 hash of the source document for caching")
    elements: list[DocumentElement] = Field(default_factory=list, description="Ordered list of extracted elements")
    full_markdown: str = Field(description="Consolidated markdown representation of the document")
    stats: dict[str, int] = Field(default_factory=dict, description="Counts of headings, tables, code blocks, etc.")
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class Chunk(BaseModel):
    """A semantic-structural chunk prepared for vector embedding and retrieval."""
    chunk_id: str = Field(description="Deterministic chunk ID, e.g. doc_p2_c5")
    parent_id: str = Field(description="Parent section or element identifier")
    doc_name: str = Field(description="Source document name, e.g. vector_database_comparison.pdf")
    page_number: int = Field(description="Page number for accurate RAG citation")
    section_path: str = Field(description="Full hierarchical section breadcrumb trail")
    element_type: str = Field(description="Type of content: 'table', 'code', 'prose', or 'composite'")
    text: str = Field(description="Contextually enriched text used for embedding and LLM prompt context")
    raw_content: str = Field(description="Original un-prefixed content for clean user citation display")
    token_count: int = Field(description="Estimated or measured token count")
    char_count: int = Field(description="Character length of raw content")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Supplementary chunk metadata")


class ChunkManifest(BaseModel):
    """Metadata summary of a generated chunk collection."""
    total_chunks: int
    total_tokens: int
    avg_tokens_per_chunk: float
    chunks_by_type: dict[str, int]
    chunks_by_doc: dict[str, int]
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
