"""Embedder module supporting Google Gemini embeddings with automatic local fallback."""
import logging
import time
from typing import Literal
import numpy as np

from src.config import embedding_config, EmbeddingConfig

logger = logging.getLogger(__name__)


class LocalSentenceTransformerEmbedder:
    """Offline, local sentence-transformers embedder (all-MiniLM-L6-v2)."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        from sentence_transformers import SentenceTransformer
        logger.info("Initializing local embedding model: %s", model_name)
        self.model = SentenceTransformer(model_name)
        self.dimension = 384

    def embed_texts(self, texts: list[str], **kwargs) -> np.ndarray:
        """Embed list of texts with L2 normalization."""
        vecs = self.model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return vecs.astype(np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        """Embed single query with L2 normalization."""
        return self.embed_texts([query])[0]


class GeminiEmbedder:
    """Generates dense vector embeddings using Google Gemini API."""

    def __init__(self, config: EmbeddingConfig | None = None) -> None:
        self.config = config or embedding_config
        if not self.config.gemini_api_key:
            raise ValueError("GEMINI_API_KEY is not set.")
        import google.generativeai as genai
        genai.configure(api_key=self.config.gemini_api_key)
        self._genai = genai
        self.model_name = self.config.gemini_embedding_model
        self.dimension = 3072

    def embed_texts(
        self,
        texts: list[str],
        task_type: Literal["retrieval_document", "retrieval_query"] = "retrieval_document",
        batch_size: int = 15,
    ) -> np.ndarray:
        """Embed list of texts in batches with rate-limiting and backoff."""
        all_embeddings: list[list[float]] = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            retries = 3
            backoff = 2.0

            for attempt in range(retries):
                try:
                    res = self._genai.embed_content(
                        model=self.model_name,
                        content=batch,
                        task_type=task_type,
                    )
                    batch_vecs = res["embedding"]
                    if batch_vecs and isinstance(batch_vecs[0], float):
                        batch_vecs = [batch_vecs]
                    all_embeddings.extend(batch_vecs)
                    break
                except Exception as e:
                    if attempt < retries - 1:
                        logger.warning(
                            "Gemini embedding batch failed (attempt %d/%d): %s. Backoff %.1fs...",
                            attempt + 1,
                            retries,
                            e,
                            backoff,
                        )
                        time.sleep(backoff)
                        backoff *= 2
                    else:
                        raise

            time.sleep(0.5)  # Respect free tier rate limits

        arr = np.array(all_embeddings, dtype=np.float32)
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms[norms == 0] = 1e-10
        return arr / norms

    def embed_query(self, query: str) -> np.ndarray:
        """Embed query using retrieval_query task type."""
        res = self.embed_texts([query], task_type="retrieval_query")
        return res[0]


class UnifiedEmbedder:
    """Unified embedder supporting both Gemini and local sentence-transformers."""

    def __init__(
        self,
        provider: Literal["gemini", "local", "auto"] = "auto",
        config: EmbeddingConfig | None = None,
    ) -> None:
        self.config = config or embedding_config
        self.provider = provider
        self._active_embedder = None
        self._initialize_embedder()

    def _initialize_embedder(self) -> None:
        if self.provider == "gemini":
            try:
                self._active_embedder = GeminiEmbedder(self.config)
                self.dimension = self._active_embedder.dimension
                logger.info("Using Gemini Embedder (3072 dims)")
                return
            except Exception as e:
                logger.warning("Failed to initialize Gemini embedder: %s", e)
                raise

        elif self.provider == "local":
            self._active_embedder = LocalSentenceTransformerEmbedder(self.config.fallback_local_model)
            self.dimension = self._active_embedder.dimension
            logger.info("Using Local Sentence-Transformers Embedder (%d dims)", self.dimension)
            return

        # Auto mode: try Gemini, fall back to local if quota exceeded or missing key
        if self.config.gemini_api_key:
            try:
                gem = GeminiEmbedder(self.config)
                # Quick test ping
                gem.embed_query("ping")
                self._active_embedder = gem
                self.provider = "gemini"
                self.dimension = gem.dimension
                logger.info("Auto-selected Gemini Embedder (3072 dims)")
                return
            except Exception as e:
                logger.warning("Gemini API test ping failed (%s). Falling back to local embedder.", e)

        self._active_embedder = LocalSentenceTransformerEmbedder(self.config.fallback_local_model)
        self.provider = "local"
        self.dimension = self._active_embedder.dimension
        logger.info("Auto-selected Local Sentence-Transformers Embedder (%d dims)", self.dimension)

    def embed_texts(self, texts: list[str], **kwargs) -> np.ndarray:
        """Embed list of texts using active embedder with fallback."""
        try:
            return self._active_embedder.embed_texts(texts, **kwargs)
        except Exception as e:
            if self.provider == "gemini":
                logger.warning("Gemini embedding failed with error: %s. Switching to local embedder...", e)
                self._active_embedder = LocalSentenceTransformerEmbedder(self.config.fallback_local_model)
                self.provider = "local"
                self.dimension = self._active_embedder.dimension
                return self._active_embedder.embed_texts(texts)
            raise

    def embed_query(self, query: str) -> np.ndarray:
        """Embed single query string."""
        try:
            return self._active_embedder.embed_query(query)
        except Exception as e:
            if self.provider == "gemini":
                logger.warning("Gemini query embedding failed: %s. Switching to local embedder...", e)
                self._active_embedder = LocalSentenceTransformerEmbedder(self.config.fallback_local_model)
                self.provider = "local"
                self.dimension = self._active_embedder.dimension
                return self._active_embedder.embed_query(query)
            raise
