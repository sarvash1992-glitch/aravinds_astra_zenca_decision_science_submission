"""CLI runner script for semantic-structural chunking pipeline."""
import argparse
import json
import logging
import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import EXTRACTED_DATA_DIR, CHUNKS_DATA_DIR
from src.ingestion.chunker import HybridSemanticStructuralChunker
from src.ingestion.models import Chunk, ChunkManifest, ExtractedDocument


def setup_logging(verbose: bool = False) -> None:
    """Configure structured logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def main() -> None:
    """Run chunking across all extracted JSON files."""
    parser = argparse.ArgumentParser(
        description="Chunk extracted documents using Hybrid Semantic-Structural strategy."
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default=str(EXTRACTED_DATA_DIR),
        help=f"Directory containing extracted JSON files (default: {EXTRACTED_DATA_DIR})",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(CHUNKS_DATA_DIR),
        help=f"Directory to save chunk JSON files (default: {CHUNKS_DATA_DIR})",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose debug logging",
    )

    args = parser.parse_args()
    setup_logging(args.verbose)

    input_path = Path(args.input_dir).resolve()
    output_path = Path(args.output_dir).resolve()
    output_path.mkdir(parents=True, exist_ok=True)

    extracted_files = sorted(list(input_path.glob("*.json")))
    # Exclude manifest.json
    extracted_files = [f for f in extracted_files if f.name != "manifest.json"]

    if not extracted_files:
        print(f"No extracted document JSON files found in {input_path}")
        sys.exit(1)

    print("=" * 60)
    print(" Hybrid Semantic-Structural Chunking Pipeline")
    print("=" * 60)
    print(f"Input directory : {input_path}")
    print(f"Output directory: {output_path}")
    print(f"Found {len(extracted_files)} documents to chunk.")
    print("-" * 60)

    chunker = HybridSemanticStructuralChunker()
    all_chunks: list[Chunk] = []
    chunks_by_doc: dict[str, int] = {}
    chunks_by_type: dict[str, int] = {}
    total_tokens = 0

    for json_file in extracted_files:
        with open(json_file, "r", encoding="utf-8") as f:
            doc_data = json.load(f)
        doc = ExtractedDocument.model_validate(doc_data)

        doc_chunks = chunker.chunk_document(doc)
        all_chunks.extend(doc_chunks)
        chunks_by_doc[doc.doc_name] = len(doc_chunks)

        # Save per-document chunk file
        doc_chunks_file = output_path / f"{json_file.stem}_chunks.json"
        with open(doc_chunks_file, "w", encoding="utf-8") as f:
            f.write(json.dumps([c.model_dump() for c in doc_chunks], indent=2, ensure_ascii=False))

        for c in doc_chunks:
            total_tokens += c.token_count
            chunks_by_type[c.element_type] = chunks_by_type.get(c.element_type, 0) + 1

        print(f" - {doc.doc_name:35}: {len(doc_chunks)} chunks (avg {sum(c.token_count for c in doc_chunks)/max(1, len(doc_chunks)):.1f} tokens)")

    # Save consolidated all_chunks.json
    all_chunks_file = output_path / "all_chunks.json"
    with open(all_chunks_file, "w", encoding="utf-8") as f:
        f.write(json.dumps([c.model_dump() for c in all_chunks], indent=2, ensure_ascii=False))

    avg_tokens = round(total_tokens / max(1, len(all_chunks)), 1)
    manifest = ChunkManifest(
        total_chunks=len(all_chunks),
        total_tokens=total_tokens,
        avg_tokens_per_chunk=avg_tokens,
        chunks_by_type=chunks_by_type,
        chunks_by_doc=chunks_by_doc,
    )

    manifest_file = output_path / "manifest.json"
    with open(manifest_file, "w", encoding="utf-8") as f:
        f.write(manifest.model_dump_json(indent=2))

    print("-" * 60)
    print(" Chunking Summary")
    print("-" * 60)
    print(f"Total Chunks        : {len(all_chunks)}")
    print(f"Total Tokens        : {total_tokens}")
    print(f"Average Tokens/Chunk: {avg_tokens}")
    print(f"Chunks by Type      : {chunks_by_type}")
    print(f"All chunks saved to : {all_chunks_file}")
    print(f"Manifest written to : {manifest_file}")
    print("=" * 60)


if __name__ == "__main__":
    main()
