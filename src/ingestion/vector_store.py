"""Hybrid Vector Store combining FAISS dense vector search and BM25 sparse retrieval."""
import json
import logging
import pickle
import re
from pathlib import Path
from typing import Any

import faiss
import numpy as np
from rank_bm25 import BM25Okapi

from src.config import embedding_config, EmbeddingConfig
from src.ingestion.embedder import UnifiedEmbedder
from src.ingestion.models import Chunk

logger = logging.getLogger(__name__)


class SearchResult:
    """Represents a retrieved chunk with similarity score and metadata."""

    def __init__(
        self,
        chunk: Chunk,
        score: float,
        retrieval_method: str = "hybrid",
        dense_rank: int | None = None,
        sparse_rank: int | None = None,
    ) -> None:
        self.chunk = chunk
        self.score = score
        self.retrieval_method = retrieval_method
        self.dense_rank = dense_rank
        self.sparse_rank = sparse_rank

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk.chunk_id,
            "doc_name": self.chunk.doc_name,
            "page_number": self.chunk.page_number,
            "section_path": self.chunk.section_path,
            "element_type": self.chunk.element_type,
            "score": round(self.score, 4),
            "retrieval_method": self.retrieval_method,
            "text": self.chunk.text,
            "raw_content": self.chunk.raw_content,
        }


class HybridVectorStore:
    """Production-grade hybrid search index with FAISS (dense) + BM25 (sparse) + RRF."""

    def __init__(
        self,
        config: EmbeddingConfig | None = None,
        embedder: UnifiedEmbedder | None = None,
    ) -> None:
        self.config = config or embedding_config
        self.embedder = embedder or UnifiedEmbedder(config=self.config)
        self.chunks: list[Chunk] = []
        self.faiss_index: faiss.IndexFlatIP | None = None
        self.bm25_index: BM25Okapi | None = None
        self._tokenized_corpus: list[list[str]] = []

    def _tokenize(self, text: str) -> list[str]:
        """Simple alphanumeric tokenizer for BM25."""
        return re.findall(r"\w+", text.lower())

    def build_index(self, chunks: list[Chunk]) -> None:
        """Build FAISS dense index and BM25 sparse index from chunk collection.
        
        Args:
            chunks: List of Chunk objects to index.
        """
        if not chunks:
            raise ValueError("No chunks provided to index.")

        self.chunks = chunks
        logger.info("Building hybrid index for %d chunks...", len(chunks))

        # 1. Build FAISS dense index
        texts_to_embed = [c.text for c in chunks]
        logger.info("Generating Gemini embeddings for %d chunks...", len(texts_to_embed))
        embeddings = self.embedder.embed_texts(texts_to_embed, task_type="retrieval_document")

        dim = embeddings.shape[1]
        self.faiss_index = faiss.IndexFlatIP(dim)
        self.faiss_index.add(embeddings)
        logger.info("FAISS index built with %d vectors (dimension: %d)", self.faiss_index.ntotal, dim)

        # 2. Build BM25 sparse index
        self._tokenized_corpus = [self._tokenize(c.text) for c in chunks]
        self.bm25_index = BM25Okapi(self._tokenized_corpus)
        logger.info("BM25 index built with %d documents", len(self._tokenized_corpus))

    def save(self, directory: Path | str | None = None) -> None:
        """Persist indices and chunk metadata to disk."""
        target_dir = Path(directory or self.config.vector_store_dir).resolve()
        target_dir.mkdir(parents=True, exist_ok=True)

        if self.faiss_index is None or self.bm25_index is None:
            raise RuntimeError("Indices not built. Call build_index() before save().")

        # 1. Save FAISS index
        faiss_path = target_dir / "index.faiss"
        faiss.write_index(self.faiss_index, str(faiss_path))

        # 2. Save BM25 index and corpus
        bm25_path = target_dir / "bm25.pkl"
        with open(bm25_path, "wb") as f:
            pickle.dump({"index": self.bm25_index, "corpus": self._tokenized_corpus}, f)

        # 3. Save chunks catalog
        chunks_path = target_dir / "chunks.json"
        with open(chunks_path, "w", encoding="utf-8") as f:
            json.dump([c.model_dump() for c in self.chunks], f, indent=2, ensure_ascii=False)

        logger.info("Hybrid vector store saved to %s", target_dir)

    def load(self, directory: Path | str | None = None) -> None:
        """Load persisted indices and chunk metadata from disk."""
        target_dir = Path(directory or self.config.vector_store_dir).resolve()
        faiss_path = target_dir / "index.faiss"
        bm25_path = target_dir / "bm25.pkl"
        chunks_path = target_dir / "chunks.json"

        if not (faiss_path.is_file() and bm25_path.is_file() and chunks_path.is_file()):
            raise FileNotFoundError(f"Vector store artifacts not found in {target_dir}")

        # 1. Load FAISS index
        self.faiss_index = faiss.read_index(str(faiss_path))

        # Auto-align embedder to match loaded index dimension
        if self.embedder.dimension != self.faiss_index.d:
            logger.info(
                "Loaded FAISS dimension (%d) differs from embedder dimension (%d). Aligning embedder...",
                self.faiss_index.d,
                self.embedder.dimension,
            )
            if self.faiss_index.d == 384:
                self.embedder = UnifiedEmbedder(provider="local", config=self.config)
            else:
                self.embedder = UnifiedEmbedder(provider="gemini", config=self.config)

        # 2. Load BM25 index
        with open(bm25_path, "rb") as f:
            bm25_data = pickle.load(f)
            self.bm25_index = bm25_data["index"]
            self._tokenized_corpus = bm25_data["corpus"]

        # 3. Load chunks catalog
        with open(chunks_path, "r", encoding="utf-8") as f:
            chunks_data = json.load(f)
            self.chunks = [Chunk.model_validate(c) for c in chunks_data]

        logger.info("Loaded hybrid vector store from %s (%d chunks)", target_dir, len(self.chunks))

    def search_dense(self, query: str, top_k: int = 5) -> list[tuple[int, float]]:
        """Search using FAISS dense vector similarity."""
        if self.faiss_index is None:
            raise RuntimeError("FAISS index is not initialized.")
        query_vec = self.embedder.embed_query(query).reshape(1, -1)
        scores, indices = self.faiss_index.search(query_vec, top_k)
        return [(int(idx), float(score)) for idx, score in zip(indices[0], scores[0]) if idx >= 0]

    def search_sparse(self, query: str, top_k: int = 5) -> list[tuple[int, float]]:
        """Search using BM25 keyword matching."""
        if self.bm25_index is None:
            raise RuntimeError("BM25 index is not initialized.")
        tokenized_query = self._tokenize(query)
        doc_scores = self.bm25_index.get_scores(tokenized_query)
        top_indices = np.argsort(doc_scores)[::-1][:top_k]
        return [(int(idx), float(doc_scores[idx])) for idx in top_indices if doc_scores[idx] > 0]

    def search_hybrid(
        self,
        query: str,
        top_k: int = 5,
        rrf_k: int = 60,
        doc_filter: str | None = None,
    ) -> list[SearchResult]:
        """Perform hybrid search combining dense & sparse results via Reciprocal Rank Fusion (RRF).
        
        Formula: RRF_score(d) = 1/(k + rank_dense(d)) + 1/(k + rank_sparse(d))
        (As detailed in RAG Architecture Patterns Section 4 & 6)
        """
        fetch_k = min(len(self.chunks), top_k * 3)

        dense_results = self.search_dense(query, top_k=fetch_k)
        sparse_results = self.search_sparse(query, top_k=fetch_k)

        # Build rank maps
        dense_rank_map = {idx: rank + 1 for rank, (idx, _) in enumerate(dense_results)}
        sparse_rank_map = {idx: rank + 1 for rank, (idx, _) in enumerate(sparse_results)}

        all_candidate_indices = set(dense_rank_map.keys()) | set(sparse_rank_map.keys())

        scored_candidates: list[tuple[int, float]] = []
        for idx in all_candidate_indices:
            chunk = self.chunks[idx]
            # Optional metadata filter
            if doc_filter and doc_filter.lower() not in chunk.doc_name.lower():
                continue

            r_dense = dense_rank_map.get(idx)
            r_sparse = sparse_rank_map.get(idx)

            # Reciprocal Rank Fusion calculation
            score = 0.0
            if r_dense is not None:
                score += 1.0 / (rrf_k + r_dense)
            if r_sparse is not None:
                score += 1.0 / (rrf_k + r_sparse)

            scored_candidates.append((idx, score))

        # Sort by fused score descending
        scored_candidates.sort(key=lambda x: x[1], reverse=True)

        results: list[SearchResult] = []
        for idx, score in scored_candidates[:top_k]:
            results.append(
                SearchResult(
                    chunk=self.chunks[idx],
                    score=score,
                    retrieval_method="hybrid_rrf",
                    dense_rank=dense_rank_map.get(idx),
                    sparse_rank=sparse_rank_map.get(idx),
                )
            )

        return results
