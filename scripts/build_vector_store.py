"""CLI runner script to build and persist the hybrid vector store."""
import argparse
import json
import logging
import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import CHUNKS_DATA_DIR, VECTOR_STORE_DIR
from src.ingestion.models import Chunk
from src.ingestion.vector_store import HybridVectorStore


def setup_logging(verbose: bool = False) -> None:
    """Configure structured logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def main() -> None:
    """Index chunks into FAISS and BM25 hybrid vector store."""
    parser = argparse.ArgumentParser(
        description="Build and persist the Hybrid (FAISS + BM25) vector store."
    )
    parser.add_argument(
        "--chunks-file",
        type=str,
        default=str(CHUNKS_DATA_DIR / "all_chunks.json"),
        help="Path to all_chunks.json file",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(VECTOR_STORE_DIR),
        help="Directory to persist vector store artifacts",
    )
    parser.add_argument(
        "--provider",
        type=str,
        choices=["auto", "local", "gemini"],
        default="auto",
        help="Embedding provider: 'auto' (Gemini with local fallback), 'local' (MiniLM), or 'gemini'",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable debug logging",
    )

    args = parser.parse_args()
    setup_logging(args.verbose)

    chunks_path = Path(args.chunks_file).resolve()
    output_path = Path(args.output_dir).resolve()

    if not chunks_path.is_file():
        print(f"Error: Chunks file not found at {chunks_path}")
        print("Please run 'python scripts/run_chunking.py' first.")
        sys.exit(1)

    print("=" * 60)
    print(" Hybrid Vector Store Indexing (FAISS + BM25)")
    print("=" * 60)
    print(f"Chunks input : {chunks_path}")
    print(f"Index output : {output_path}")
    print(f"Provider     : {args.provider}")

    with open(chunks_path, "r", encoding="utf-8") as f:
        chunks_data = json.load(f)
    chunks = [Chunk.model_validate(c) for c in chunks_data]
    print(f"Loaded {len(chunks)} chunks to index.")
    print("-" * 60)

    from src.ingestion.embedder import UnifiedEmbedder
    embedder = UnifiedEmbedder(provider=args.provider)
    store = HybridVectorStore(embedder=embedder)
    store.build_index(chunks)
    store.save(output_path)

    print("-" * 60)
    print(" Verifying Hybrid Retrieval with Test Query")
    print("-" * 60)
    test_query = "Which vector database has the lowest latency at 1M vectors?"
    print(f"Query: '{test_query}'\n")

    results = store.search_hybrid(test_query, top_k=3)
    for i, res in enumerate(results, 1):
        print(f"[{i}] Score: {res.score:.4f} | Source: {res.chunk.doc_name} (p.{res.chunk.page_number})")
        print(f"    Section: {res.chunk.section_path}")
        snippet = res.chunk.raw_content[:180].replace("\n", " ")
        print(f"    Snippet: {snippet}...\n")

    print("=" * 60)
    print(" Indexing Complete! Hybrid vector store ready for RAG.")
    print("=" * 60)


if __name__ == "__main__":
    main()
