"""Ingestion pipeline for batch extraction and persistence of documents."""
import json
import logging
import time
from pathlib import Path
from typing import Any

from src.config import extraction_config, ExtractionConfig
from src.ingestion.extractor import HybridDocumentExtractor
from src.ingestion.models import ExtractedDocument

logger = logging.getLogger(__name__)


class ExtractionPipeline:
    """Batch document extraction pipeline for the RAG knowledge base.
    
    Orchestrates:
    - Scanning input directory for PDF documents
    - Executing HybridDocumentExtractor on each document
    - Persisting structured JSON with element-level metadata
    - Persisting human-readable Markdown documents
    - Creating an extraction manifest with checksums and metrics
    """

    def __init__(self, config: ExtractionConfig | None = None) -> None:
        self.config = config or extraction_config
        self.extractor = HybridDocumentExtractor(self.config)

    def run(
        self,
        input_dir: Path | str | None = None,
        output_dir: Path | str | None = None,
        force_reextract: bool = False,
    ) -> dict[str, Any]:
        """Execute extraction across all PDF files in the input directory.
        
        Args:
            input_dir: Directory containing source PDFs (default: RAG_DOCS_DIR).
            output_dir: Directory to save extracted artifacts (default: EXTRACTED_DATA_DIR).
            force_reextract: If True, re-extracts even if cached output exists.
            
        Returns:
            Dictionary containing overall extraction manifest and status.
        """
        in_path = Path(input_dir or self.config.rag_docs_dir).resolve()
        out_path = Path(output_dir or self.config.extracted_data_dir).resolve()
        out_path.mkdir(parents=True, exist_ok=True)

        pdf_files = sorted(list(in_path.glob("*.pdf")))
        if not pdf_files:
            logger.warning("No PDF documents found in %s", in_path)
            return {"status": "empty", "documents": [], "total_documents": 0}

        logger.info("Found %d PDF documents to process in %s", len(pdf_files), in_path)
        start_time = time.time()
        
        processed_docs: list[dict[str, Any]] = []
        total_pages = 0
        total_elements = 0
        total_tables = 0
        total_code_blocks = 0

        for pdf_file in pdf_files:
            file_stem = pdf_file.stem
            json_target = out_path / f"{file_stem}.json"
            md_target = out_path / f"{file_stem}.md"

            current_hash = self.extractor.compute_file_hash(pdf_file)

            # Caching check: skip if existing json has identical hash
            if not force_reextract and json_target.is_file() and md_target.is_file():
                try:
                    with open(json_target, "r", encoding="utf-8") as f:
                        cached_data = json.load(f)
                    if cached_data.get("file_hash") == current_hash:
                        logger.info("Skipping %s (cached output is up-to-date)", pdf_file.name)
                        stats = cached_data.get("stats", {})
                        total_pages += cached_data.get("total_pages", 0)
                        total_elements += len(cached_data.get("elements", []))
                        total_tables += stats.get("tables", 0)
                        total_code_blocks += stats.get("code_blocks", 0)
                        processed_docs.append({
                            "doc_name": cached_data.get("doc_name"),
                            "status": "cached",
                            "pages": cached_data.get("total_pages"),
                            "elements": len(cached_data.get("elements", [])),
                            "stats": stats,
                            "json_path": str(json_target),
                            "md_path": str(md_target),
                        })
                        continue
                except Exception as e:
                    logger.warning("Error reading cache for %s: %s. Re-extracting...", pdf_file.name, e)

            # Perform extraction
            doc_start = time.time()
            extracted_doc = self.extractor.extract_document(pdf_file)
            doc_duration = time.time() - doc_start

            # Save structured JSON
            with open(json_target, "w", encoding="utf-8") as f:
                f.write(extracted_doc.model_dump_json(indent=2))

            # Save clean Markdown
            with open(md_target, "w", encoding="utf-8") as f:
                f.write(extracted_doc.full_markdown)

            total_pages += extracted_doc.total_pages
            total_elements += len(extracted_doc.elements)
            total_tables += extracted_doc.stats.get("tables", 0)
            total_code_blocks += extracted_doc.stats.get("code_blocks", 0)

            logger.info(
                "Extracted %s in %.2fs: %d pages, %d elements (%d tables, %d code blocks)",
                pdf_file.name,
                doc_duration,
                extracted_doc.total_pages,
                len(extracted_doc.elements),
                extracted_doc.stats.get("tables", 0),
                extracted_doc.stats.get("code_blocks", 0),
            )

            processed_docs.append({
                "doc_name": extracted_doc.doc_name,
                "status": "extracted",
                "duration_seconds": round(doc_duration, 2),
                "pages": extracted_doc.total_pages,
                "elements": len(extracted_doc.elements),
                "stats": extracted_doc.stats,
                "json_path": str(json_target),
                "md_path": str(md_target),
            })

        total_duration = time.time() - start_time
        manifest = {
            "status": "success",
            "total_documents": len(pdf_files),
            "total_pages": total_pages,
            "total_elements": total_elements,
            "total_tables": total_tables,
            "total_code_blocks": total_code_blocks,
            "duration_seconds": round(total_duration, 2),
            "input_dir": str(in_path),
            "output_dir": str(out_path),
            "documents": processed_docs,
        }

        # Save manifest
        manifest_file = out_path / "manifest.json"
        with open(manifest_file, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

        logger.info("Extraction completed in %.2fs. Manifest written to %s", total_duration, manifest_file)
        return manifest
