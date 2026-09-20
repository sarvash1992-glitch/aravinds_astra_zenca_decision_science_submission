"""Hybrid Document Extractor combining PyMuPDF layout analysis and pdfplumber table extraction."""
import hashlib
import logging
import re
from pathlib import Path
from typing import Any

import fitz  # PyMuPDF
import pdfplumber

from src.config import extraction_config, ExtractionConfig
from src.ingestion.models import (
    DocumentElement,
    ElementType,
    ExtractedDocument,
    ExtractedTable,
)

logger = logging.getLogger(__name__)


class HybridDocumentExtractor:
    """Production-ready hybrid document extractor.
    
    Combines:
    1. PyMuPDF span-level layout inspection (font styles, code blocks, headings, reading order)
    2. pdfplumber grid line detection for explicit border tables
    3. Structural block & column-aligned matrix recognition for multi-column comparison tables
    4. Monospace font recognition for Python code, tool schemas, and formulas
    5. Header/footer filtration and hierarchical section propagation.
    """

    def __init__(self, config: ExtractionConfig | None = None) -> None:
        self.config = config or extraction_config
        self._header_footer_regexes = [
            re.compile(pat, re.IGNORECASE) for pat in self.config.header_footer_patterns
        ]

    def compute_file_hash(self, file_path: Path) -> str:
        """Compute SHA-256 hash of the input file."""
        hasher = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                hasher.update(chunk)
        return hasher.hexdigest()

    def is_header_or_footer(self, text: str) -> bool:
        """Check if a text string matches known running headers or footers."""
        cleaned = re.sub(r"\s+", " ", text).strip()
        if not cleaned:
            return True
        for pattern in self._header_footer_regexes:
            if pattern.search(cleaned):
                return True
        if re.match(r"^Page\s+\d+(\s+of\s+\d+)?$", cleaned, re.IGNORECASE):
            return True
        return False

    def is_code_span(self, font_name: str) -> bool:
        """Determine if a font represents monospace/code."""
        font_lower = font_name.lower()
        return any(kw in font_lower for kw in self.config.code_font_keywords)

    def is_valid_grid_table(self, table_data: list[list[Any]]) -> bool:
        """Validate if an extracted grid table contains legitimate tabular data."""
        if not table_data or len(table_data) < self.config.min_table_rows:
            return False

        total_cells = sum(len(row) for row in table_data)
        non_empty_cells = sum(
            1 for row in table_data for cell in row if cell and str(cell).strip()
        )
        if total_cells == 0 or non_empty_cells / total_cells < 0.35:
            return False

        cell_lengths = [
            len(str(cell).strip())
            for row in table_data
            for cell in row
            if cell and str(cell).strip()
        ]
        if not cell_lengths or max(cell_lengths) > 350:
            return False

        first_row = [str(c).strip() for c in table_data[0] if c and str(c).strip()]
        if len(first_row) < self.config.min_table_cols:
            return False

        return True

    def extract_grid_tables(
        self, fitz_page: fitz.Page, pdfplumber_page: pdfplumber.page.Page
    ) -> list[ExtractedTable]:
        """Extract explicit grid tables using PyMuPDF and pdfplumber."""
        extracted: list[ExtractedTable] = []

        # 1. Try PyMuPDF TableFinder
        tabs = fitz_page.find_tables()
        for t in tabs.tables:
            data = t.extract()
            if self.is_valid_grid_table(data):
                cleaned_rows = [
                    [re.sub(r"\s+", " ", str(c or "").strip()) for c in row]
                    for row in data
                ]
                headers = cleaned_rows[0]
                body_rows = cleaned_rows[1:]
                num_cols = len(headers)
                md_table = self._format_markdown_table(headers, body_rows, num_cols)
                extracted.append(
                    ExtractedTable(
                        headers=headers,
                        rows=body_rows,
                        markdown=md_table,
                        bbox=t.bbox,
                    )
                )

        # 2. Try pdfplumber lines if no valid tables from PyMuPDF
        if not extracted:
            plumb_tables = pdfplumber_page.find_tables(
                table_settings={
                    "vertical_strategy": "lines",
                    "horizontal_strategy": "lines",
                    "snap_tolerance": self.config.table_snap_tolerance,
                }
            )
            for pt in plumb_tables:
                data = pt.extract()
                if self.is_valid_grid_table(data):
                    cleaned_rows = [
                        [re.sub(r"\s+", " ", str(c or "").strip()) for c in row]
                        for row in data
                    ]
                    headers = cleaned_rows[0]
                    body_rows = cleaned_rows[1:]
                    num_cols = len(headers)
                    md_table = self._format_markdown_table(headers, body_rows, num_cols)
                    extracted.append(
                        ExtractedTable(
                            headers=headers,
                            rows=body_rows,
                            markdown=md_table,
                            bbox=pt.bbox,
                        )
                    )

        return extracted

    def _format_markdown_table(
        self, headers: list[str], rows: list[list[str]], num_cols: int
    ) -> str:
        """Format rows and headers into GitHub Flavored Markdown table."""
        safe_headers = [h.replace("|", "\\|") if h else f"Col {i+1}" for i, h in enumerate(headers)]
        header_line = "| " + " | ".join(safe_headers) + " |"
        sep_line = "| " + " | ".join(["---"] * num_cols) + " |"
        row_lines = []
        for row in rows:
            padded_row = (row + [""] * num_cols)[:num_cols]
            safe_row = [c.replace("|", "\\|") for c in padded_row]
            row_lines.append("| " + " | ".join(safe_row) + " |")
        return "\n".join([header_line, sep_line] + row_lines)

    def is_inside_bboxes(
        self, point_bbox: tuple[float, float, float, float], table_bboxes: list[tuple[float, float, float, float]]
    ) -> bool:
        """Check if a bounding box overlaps significantly with any table bounding box."""
        bx0, by0, bx1, by1 = point_bbox
        for tx0, ty0, tx1, ty1 in table_bboxes:
            x_overlap = max(0, min(bx1, tx1) - max(bx0, tx0))
            y_overlap = max(0, min(by1, ty1) - max(by0, ty0))
            overlap_area = x_overlap * y_overlap
            box_area = max(1e-5, (bx1 - bx0) * (by1 - by0))
            if overlap_area / box_area > 0.4:
                return True
        return False

    def extract_document(self, pdf_path: Path | str) -> ExtractedDocument:
        """Extract structured content from a PDF document.
        
        Args:
            pdf_path: Path to the PDF file.
            
        Returns:
            ExtractedDocument containing ordered elements and full markdown.
        """
        path = Path(pdf_path).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"PDF document not found: {path}")

        file_hash = self.compute_file_hash(path)
        doc_name = path.name

        fitz_doc = fitz.open(path)
        total_pages = len(fitz_doc)
        
        logger.info("Extracting %s (%d pages)", doc_name, total_pages)

        elements: list[DocumentElement] = []
        element_counter = 0
        current_section = "Introduction"
        stats = {
            "headings": 0,
            "paragraphs": 0,
            "tables": 0,
            "code_blocks": 0,
        }

        with pdfplumber.open(path) as plumber_doc:
            for page_idx in range(total_pages):
                page_num = page_idx + 1
                fitz_page = fitz_doc[page_idx]
                plumber_page = plumber_doc.pages[page_idx]

                # 1. Extract valid grid tables first
                grid_tables = self.extract_grid_tables(fitz_page, plumber_page)
                grid_table_bboxes = [t.bbox for t in grid_tables if t.bbox]

                # 2. Get text blocks from PyMuPDF
                raw_blocks = fitz_page.get_text("blocks")
                blocks = []
                for b in raw_blocks:
                    if b[6] == 0:  # text block
                        blocks.append(b)

                # Sort blocks top-to-bottom
                blocks.sort(key=lambda b: (round(b[1] / 5.0) * 5.0, b[0]))

                i = 0
                emitted_grid_indices = set()

                while i < len(blocks):
                    b = blocks[i]
                    bbox = (b[0], b[1], b[2], b[3])
                    raw_text = b[4].strip()

                    # Check if inside already detected grid table
                    if self.is_inside_bboxes(bbox, grid_table_bboxes):
                        for g_idx, g_tbl in enumerate(grid_tables):
                            if g_idx not in emitted_grid_indices and g_tbl.bbox and self.is_inside_bboxes(bbox, [g_tbl.bbox]):
                                element_counter += 1
                                elements.append(
                                    DocumentElement(
                                        id=f"{path.stem}_p{page_num}_e{element_counter}",
                                        page_number=page_num,
                                        element_type=ElementType.TABLE,
                                        content=g_tbl.markdown,
                                        section=current_section,
                                        bbox=g_tbl.bbox,
                                        metadata={
                                            "num_rows": len(g_tbl.rows),
                                            "num_cols": len(g_tbl.headers),
                                            "headers": g_tbl.headers,
                                        },
                                    )
                                )
                                stats["tables"] += 1
                                emitted_grid_indices.add(g_idx)
                        i += 1
                        continue

                    # Filter headers and footers
                    if self.is_header_or_footer(raw_text):
                        i += 1
                        continue

                    lines = [l.strip() for l in raw_text.split("\n") if l.strip()]
                    num_lines = len(lines)

                    # Look-ahead for consecutive multiline blocks with identical column count
                    # Typical for comparisons: 3 to 8 columns
                    if 3 <= num_lines <= 8 and i + 1 < len(blocks):
                        table_rows = [lines]
                        next_i = i + 1
                        while next_i < len(blocks):
                            nb = blocks[next_i]
                            n_text = nb[4].strip()
                            if self.is_header_or_footer(n_text):
                                next_i += 1
                                continue
                            n_lines = [l.strip() for l in n_text.split("\n") if l.strip()]
                            if len(n_lines) == num_lines:
                                table_rows.append(n_lines)
                                next_i += 1
                            else:
                                break

                        # If we accumulated at least 2 rows (header + 1 row)
                        if len(table_rows) >= 2:
                            element_counter += 1
                            headers = table_rows[0]
                            body = table_rows[1:]
                            md_table = self._format_markdown_table(headers, body, num_lines)
                            tbl_bbox = (b[0], b[1], blocks[next_i - 1][2], blocks[next_i - 1][3])
                            elements.append(
                                DocumentElement(
                                    id=f"{path.stem}_p{page_num}_e{element_counter}",
                                    page_number=page_num,
                                    element_type=ElementType.TABLE,
                                    content=md_table,
                                    section=current_section,
                                    bbox=tbl_bbox,
                                    metadata={
                                        "num_rows": len(body),
                                        "num_cols": num_lines,
                                        "headers": headers,
                                    },
                                )
                            )
                            stats["tables"] += 1
                            i = next_i
                            continue

                    # Inspect detailed spans for code / heading / paragraph classification
                    page_dict = fitz_page.get_text("dict", clip=fitz.Rect(bbox))
                    spans = []
                    max_font_size = 0.0
                    is_bold = False
                    mono_char_count = 0
                    total_char_count = 0
                    annotated_lines = []

                    for db in page_dict.get("blocks", []):
                        for dl in db.get("lines", []):
                            line_str = ""
                            for ds in dl.get("spans", []):
                                stext = ds.get("text", "")
                                if not stext:
                                    continue
                                fn = ds.get("font", "")
                                fsz = ds.get("size", 10.0)
                                flg = ds.get("flags", 0)
                                is_m = self.is_code_span(fn)

                                total_char_count += len(stext)
                                if is_m:
                                    mono_char_count += len(stext)
                                    # Format inline code
                                    if stext.strip() and len(stext.strip()) > 1:
                                        line_str += f"`{stext.strip()}` "
                                    else:
                                        line_str += stext
                                else:
                                    line_str += stext

                                if fsz > max_font_size:
                                    max_font_size = fsz
                                if (flg & 2) or "bold" in fn.lower():
                                    is_bold = True
                                spans.append(ds)
                            if line_str:
                                annotated_lines.append(line_str)

                    element_counter += 1
                    elem_id = f"{path.stem}_p{page_num}_e{element_counter}"

                    # Explicit code block detection:
                    # 1. >75% monospace characters, OR
                    # 2. explicit code prefix with multiple lines
                    is_full_mono_code = (
                        total_char_count > 20
                        and (mono_char_count / max(1, total_char_count)) > 0.75
                    )
                    has_explicit_code_prefix = (
                        (raw_text.startswith("# Tool schema")
                        or raw_text.startswith("def ")
                        or raw_text.startswith("class ")
                        or raw_text.startswith("import ")
                        or raw_text.startswith("# LCEL")
                        or "TOOL_SCHEMA =" in raw_text
                        or raw_text.startswith("curl "))
                        and len(lines) > 1
                    )

                    if is_full_mono_code or has_explicit_code_prefix:
                        lang = "python" if any(k in raw_text for k in ["def ", "class ", "import ", "TOOL_SCHEMA", "#"]) else "text"
                        formatted_code = f"```{lang}\n{raw_text}\n```"
                        elements.append(
                            DocumentElement(
                                id=elem_id,
                                page_number=page_num,
                                element_type=ElementType.CODE,
                                content=formatted_code,
                                section=current_section,
                                bbox=bbox,
                                metadata={"language": lang},
                            )
                        )
                        stats["code_blocks"] += 1

                    # Detect Headings
                    elif (
                        (max_font_size >= self.config.min_heading_font_size and is_bold)
                        or re.match(r"^(SECTION\s+\d+|S\s+E\s+C\s+T\s+I\s+O\s+N\s+\d+|\d+(\.\d+)*\s+[A-Z])", raw_text)
                    ):
                        clean_title = re.sub(r"\s+", " ", raw_text).strip()
                        if max_font_size >= 14.0 or "SECTION" in clean_title.upper():
                            heading_md = f"# {clean_title}"
                            current_section = clean_title
                        elif max_font_size >= 12.0:
                            heading_md = f"## {clean_title}"
                            current_section = f"{current_section} > {clean_title}"
                        else:
                            heading_md = f"### {clean_title}"
                            current_section = f"{current_section} > {clean_title}"

                        elements.append(
                            DocumentElement(
                                id=elem_id,
                                page_number=page_num,
                                element_type=ElementType.HEADING,
                                content=heading_md,
                                section=current_section,
                                bbox=bbox,
                                metadata={"level": heading_md.count("#"), "font_size": max_font_size},
                            )
                        )
                        stats["headings"] += 1

                    # Normal Paragraph
                    else:
                        # Use annotated lines (with inline code backticks) if available
                        if annotated_lines:
                            content_text = " ".join(annotated_lines)
                        else:
                            content_text = raw_text
                        norm_paragraph = re.sub(r"[ \t]+", " ", content_text).strip()
                        elements.append(
                            DocumentElement(
                                id=elem_id,
                                page_number=page_num,
                                element_type=ElementType.PARAGRAPH,
                                content=norm_paragraph,
                                section=current_section,
                                bbox=bbox,
                                metadata={"font_size": max_font_size},
                            )
                        )
                        stats["paragraphs"] += 1

                    i += 1

                # If any grid table remained unemitted
                for g_idx, g_tbl in enumerate(grid_tables):
                    if g_idx not in emitted_grid_indices:
                        element_counter += 1
                        elements.append(
                            DocumentElement(
                                id=f"{path.stem}_p{page_num}_e{element_counter}",
                                page_number=page_num,
                                element_type=ElementType.TABLE,
                                content=g_tbl.markdown,
                                section=current_section,
                                bbox=g_tbl.bbox,
                                metadata={
                                    "num_rows": len(g_tbl.rows),
                                    "num_cols": len(g_tbl.headers),
                                    "headers": g_tbl.headers,
                                },
                            )
                        )
                        stats["tables"] += 1

        fitz_doc.close()

        # Build full consolidated markdown representation
        markdown_sections: list[str] = [f"# {path.stem}\n"]
        current_page = 0
        for elem in elements:
            if elem.page_number != current_page:
                current_page = elem.page_number
                markdown_sections.append(f"\n<!-- Page {current_page} -->\n")
            markdown_sections.append(f"{elem.content}\n")

        full_markdown = "\n".join(markdown_sections)

        return ExtractedDocument(
            doc_name=doc_name,
            source_path=str(path),
            total_pages=total_pages,
            file_hash=file_hash,
            elements=elements,
            full_markdown=full_markdown,
            stats=stats,
        )
