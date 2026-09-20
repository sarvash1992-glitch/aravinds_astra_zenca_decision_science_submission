"""CLI runner script for the one-time data extraction pipeline."""
import argparse
import json
import logging
import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ingestion.pipeline import ExtractionPipeline
from src.config import RAG_DOCS_DIR, EXTRACTED_DATA_DIR


def setup_logging(verbose: bool = False) -> None:
    """Configure structured logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def main() -> None:
    """Parse CLI arguments and run extraction pipeline."""
    parser = argparse.ArgumentParser(
        description="One-time data extraction pipeline for RAG knowledge documents."
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default=str(RAG_DOCS_DIR),
        help=f"Directory containing PDF documents (default: {RAG_DOCS_DIR})",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(EXTRACTED_DATA_DIR),
        help=f"Directory to save extracted outputs (default: {EXTRACTED_DATA_DIR})",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-extraction even if cached artifacts exist",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable debug logging",
    )

    args = parser.parse_args()
    setup_logging(args.verbose)

    print("=" * 60)
    print(" RAG Document Extraction Pipeline (One-Time Ingestion)")
    print("=" * 60)
    print(f"Input directory : {args.input_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Force re-extract: {args.force}")
    print("-" * 60)

    pipeline = ExtractionPipeline()
    manifest = pipeline.run(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        force_reextract=args.force,
    )

    print("\n" + "=" * 60)
    print(" Extraction Summary")
    print("=" * 60)
    print(f"Status           : {manifest.get('status')}")
    print(f"Total Documents  : {manifest.get('total_documents')}")
    print(f"Total Pages      : {manifest.get('total_pages')}")
    print(f"Total Elements   : {manifest.get('total_elements')}")
    print(f"Total Tables     : {manifest.get('total_tables')}")
    print(f"Total Code Blocks: {manifest.get('total_code_blocks')}")
    print(f"Total Duration   : {manifest.get('duration_seconds')}s")
    print("-" * 60)
    for doc in manifest.get("documents", []):
        print(f" - {doc['doc_name']}: {doc['pages']} pages, {doc['elements']} elements, {doc.get('stats', {})}")
    print(f"\nExtracted artifacts written to: {args.output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
