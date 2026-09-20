"""Hybrid Semantic-Structural Chunker for RAG knowledge ingestion."""
import logging
import re
from pathlib import Path
from typing import Any
import tiktoken

from src.config import chunking_config, ChunkingConfig
from src.ingestion.models import (
    Chunk,
    DocumentElement,
    ElementType,
    ExtractedDocument,
)

logger = logging.getLogger(__name__)


class HybridSemanticStructuralChunker:
    """Production-grade semantic-structural document chunker.
    
    Capabilities:
    1. Content-type aware:
       - Preserves tables as atomic units or splits by row blocks with repeated markdown headers.
       - Preserves code blocks intact.
       - Recursively splits long prose on semantic boundaries (\n\n, \n, sentence endings).
    2. Contextual Enrichment:
       - Prepends hierarchical document, section, and page breadcrumbs to every chunk.
    3. Token Budgeting:
       - Uses tiktoken (cl100k_base) to maintain precise token counts.
    """

    def __init__(self, config: ChunkingConfig | None = None) -> None:
        self.config = config or chunking_config
        try:
            self._tokenizer = tiktoken.get_encoding("cl100k_base")
        except Exception:
            self._tokenizer = None

    def count_tokens(self, text: str) -> int:
        """Calculate token count using tiktoken or fallback estimation."""
        if not text:
            return 0
        if self._tokenizer:
            return len(self._tokenizer.encode(text, disallowed_special=()))
        # Fallback estimation: ~4 chars per token
        return max(1, int(len(text) / 3.8))

    def _format_breadcrumb(self, doc_name: str, section: str, page: int) -> str:
        """Build contextual breadcrumb prefix for embedding."""
        clean_doc = doc_name.replace(".pdf", "").replace("_", " ").title()
        sec = section.strip() if section else "Overview"
        return f"[Document: {clean_doc} | Section: {sec} | Page: {page}]"

    def _split_sentences(self, text: str) -> list[str]:
        """Split text into sentences while respecting abbreviations."""
        # Split on period, question mark, or exclamation followed by space and uppercase
        raw_sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9✓✗•\-])", text)
        sentences = [s.strip() for s in raw_sentences if s.strip()]
        return sentences if sentences else [text]

    def _split_large_table(
        self, table_md: str, doc_name: str, section: str, page: int, base_id: str
    ) -> list[Chunk]:
        """Split a large markdown table across multiple chunks, repeating headers."""
        lines = [line.strip() for line in table_md.split("\n") if line.strip()]
        if len(lines) < 3:
            # Not a standard table; treat as single chunk
            return [
                self._create_chunk(
                    chunk_id=base_id,
                    doc_name=doc_name,
                    page=page,
                    section=section,
                    element_type="table",
                    content=table_md,
                )
            ]

        header_line = lines[0]
        separator_line = lines[1]
        data_rows = lines[2:]

        chunks = []
        current_rows: list[str] = []
        part_idx = 1

        for row in data_rows:
            candidate_rows = current_rows + [row]
            candidate_table = "\n".join([header_line, separator_line] + candidate_rows)
            tokens = self.count_tokens(candidate_table)

            if tokens > self.config.target_chunk_tokens and current_rows:
                part_table = "\n".join([header_line, separator_line] + current_rows)
                chunks.append(
                    self._create_chunk(
                        chunk_id=f"{base_id}_p{part_idx}",
                        doc_name=doc_name,
                        page=page,
                        section=section,
                        element_type="table",
                        content=part_table,
                        metadata={"is_split_table": True, "part": part_idx},
                    )
                )
                part_idx += 1
                current_rows = [row]
            else:
                current_rows.append(row)

        if current_rows:
            part_table = "\n".join([header_line, separator_line] + current_rows)
            suffix = f"_p{part_idx}" if part_idx > 1 else ""
            chunks.append(
                self._create_chunk(
                    chunk_id=f"{base_id}{suffix}",
                    doc_name=doc_name,
                    page=page,
                    section=section,
                    element_type="table",
                    content=part_table,
                    metadata={"is_split_table": part_idx > 1, "part": part_idx},
                )
            )

        return chunks

    def _create_chunk(
        self,
        chunk_id: str,
        doc_name: str,
        page: int,
        section: str,
        element_type: str,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> Chunk:
        """Assemble a complete Chunk instance with breadcrumbs and token counts."""
        breadcrumb = self._format_breadcrumb(doc_name, section, page)
        enriched_text = f"{breadcrumb}\n{content}" if self.config.inject_breadcrumbs else content
        token_count = self.count_tokens(enriched_text)

        safe_section = re.sub(r"[^\w\-]", "_", section.lower())[:40].strip("_")
        parent_id = f"{Path(doc_name).stem}_s_{safe_section}"

        return Chunk(
            chunk_id=chunk_id,
            parent_id=parent_id,
            doc_name=doc_name,
            page_number=page,
            section_path=section,
            element_type=element_type,
            text=enriched_text,
            raw_content=content,
            token_count=token_count,
            char_count=len(content),
            metadata=metadata or {},
        )

    def chunk_document(self, doc: ExtractedDocument) -> list[Chunk]:
        """Convert an ExtractedDocument into an ordered list of semantic chunks.
        
        Args:
            doc: ExtractedDocument instance containing elements.
            
        Returns:
            List of structured, token-bounded Chunk objects.
        """
        chunks: list[Chunk] = []
        doc_name = doc.doc_name
        doc_stem = Path(doc_name).stem

        chunk_counter = 0

        # Buffer for accumulating adjacent prose under same section
        current_text_buffer: list[str] = []
        current_section = "Introduction"
        current_page = 1

        def flush_text_buffer() -> None:
            nonlocal chunk_counter, current_text_buffer
            if not current_text_buffer:
                return

            joined_text = "\n\n".join(current_text_buffer).strip()
            current_text_buffer = []

            if not joined_text:
                return

            total_tokens = self.count_tokens(joined_text)

            # If within token limits, emit single chunk
            if total_tokens <= self.config.max_chunk_tokens:
                chunk_counter += 1
                chunks.append(
                    self._create_chunk(
                        chunk_id=f"{doc_stem}_c{chunk_counter}",
                        doc_name=doc_name,
                        page=current_page,
                        section=current_section,
                        element_type="prose",
                        content=joined_text,
                    )
                )
                return

            # Otherwise, split across sentences with overlap
            sentences = self._split_sentences(joined_text)
            window_sentences: list[str] = []
            
            for s in sentences:
                candidate_text = " ".join(window_sentences + [s])
                c_tokens = self.count_tokens(candidate_text)

                if c_tokens > self.config.target_chunk_tokens and window_sentences:
                    chunk_counter += 1
                    chunk_body = " ".join(window_sentences)
                    chunks.append(
                        self._create_chunk(
                            chunk_id=f"{doc_stem}_c{chunk_counter}",
                            doc_name=doc_name,
                            page=current_page,
                            section=current_section,
                            element_type="prose",
                            content=chunk_body,
                        )
                    )
                    # Keep last 1-2 sentences for overlap
                    overlap_sentences = []
                    overlap_tokens = 0
                    for prev_s in reversed(window_sentences):
                        s_tokens = self.count_tokens(prev_s)
                        if overlap_tokens + s_tokens <= self.config.chunk_overlap_tokens:
                            overlap_sentences.insert(0, prev_s)
                            overlap_tokens += s_tokens
                        else:
                            break
                    window_sentences = overlap_sentences + [s]
                else:
                    window_sentences.append(s)

            if window_sentences:
                chunk_counter += 1
                chunk_body = " ".join(window_sentences)
                chunks.append(
                    self._create_chunk(
                        chunk_id=f"{doc_stem}_c{chunk_counter}",
                        doc_name=doc_name,
                        page=current_page,
                        section=current_section,
                        element_type="prose",
                        content=chunk_body,
                    )
                )

        # Iterate over all document elements
        for elem in doc.elements:
            # 1. Update section & page context
            if elem.section:
                # If section changed, flush preceding text buffer
                if elem.section != current_section:
                    flush_text_buffer()
                    current_section = elem.section
            current_page = elem.page_number

            # 2. Handle Tables
            if elem.element_type == ElementType.TABLE:
                flush_text_buffer()
                chunk_counter += 1
                base_id = f"{doc_stem}_c{chunk_counter}"
                tbl_tokens = self.count_tokens(elem.content)
                if tbl_tokens > self.config.max_chunk_tokens:
                    table_chunks = self._split_large_table(
                        elem.content, doc_name, current_section, current_page, base_id
                    )
                    chunks.extend(table_chunks)
                else:
                    chunks.append(
                        self._create_chunk(
                            chunk_id=base_id,
                            doc_name=doc_name,
                            page=current_page,
                            section=current_section,
                            element_type="table",
                            content=elem.content,
                            metadata=elem.metadata,
                        )
                    )

            # 3. Handle Code Blocks
            elif elem.element_type == ElementType.CODE:
                flush_text_buffer()
                chunk_counter += 1
                chunks.append(
                    self._create_chunk(
                        chunk_id=f"{doc_stem}_c{chunk_counter}",
                        doc_name=doc_name,
                        page=current_page,
                        section=current_section,
                        element_type="code",
                        content=elem.content,
                        metadata=elem.metadata,
                    )
                )

            # 4. Handle Headings
            elif elem.element_type == ElementType.HEADING:
                flush_text_buffer()
                # Store heading text in buffer to prefix subsequent paragraphs
                current_text_buffer.append(f"### {elem.content.lstrip('#').strip()}")

            # 5. Handle Paragraphs
            elif elem.element_type == ElementType.PARAGRAPH:
                current_text_buffer.append(elem.content)
                # If buffer exceeds target tokens, flush it
                if self.count_tokens("\n\n".join(current_text_buffer)) >= self.config.target_chunk_tokens:
                    flush_text_buffer()

        # Flush any remaining text in buffer
        flush_text_buffer()

        logger.info("Generated %d chunks for %s", len(chunks), doc_name)
        return chunks
