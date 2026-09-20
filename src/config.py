"""Global configuration settings for the RAG and Agent system."""
import os
from pathlib import Path
from dotenv import load_dotenv
from pydantic import BaseModel, Field

# Load .env file
load_dotenv()

# Base directories
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAG_DOCS_DIR = PROJECT_ROOT / "RAG_documents"
DATA_DIR = PROJECT_ROOT / "data"
EXTRACTED_DATA_DIR = DATA_DIR / "extracted"
CHUNKS_DATA_DIR = DATA_DIR / "chunks"
VECTOR_STORE_DIR = DATA_DIR / "vectorstore"

# Ensure output directories exist
for d in [DATA_DIR, EXTRACTED_DATA_DIR, CHUNKS_DATA_DIR, VECTOR_STORE_DIR]:
    d.mkdir(parents=True, exist_ok=True)


class ExtractionConfig(BaseModel):
    """Configuration options for hybrid document extraction."""
    rag_docs_dir: Path = Field(default=RAG_DOCS_DIR)
    extracted_data_dir: Path = Field(default=EXTRACTED_DATA_DIR)
    
    # Text and Layout Extraction Parameters
    min_heading_font_size: float = Field(default=11.5, description="Font size threshold for headings")
    code_font_keywords: list[str] = Field(
        default=["mono", "courier", "consolas", "code", "dejavu-sans-mono", "noto-sans-mono"],
        description="Font family substrings that identify monospace code snippets"
    )
    
    # Table Detection Parameters
    table_snap_tolerance: float = Field(default=3.0, description="Tolerance for table line snapping")
    min_table_cols: int = Field(default=2, description="Minimum columns to consider a block a table")
    min_table_rows: int = Field(default=2, description="Minimum rows to consider a block a table")
    
    # Running Header/Footer Regex Patterns to strip
    header_footer_patterns: list[str] = Field(
        default=[
            r"^Page\s+\d+\s+of\s+\d+$",
            r"^Document\s+\d+\s+of\s+\d+$",
            r"^SENIOR AI ENGINEER TECHNICAL ASSESSMENT.*$",
            r"^SENIOR AI ENGINEER ASSESSMENT.*$",
            r"^S E N I O R\s+A I\s+E N G I N E E R.*$",
            r"^Assessment Assignment\s+\|\s+Page\s+\d+\s+of\s+\d+$",
            r"^Vector Database Comparison\s+\|\s+Document\s+\d+\s+of\s+\d+.*$",
            r"^Agentic AI Frameworks\s+—\s+Document\s+\d+\s+of\s+\d+.*$",
        ],
        description="Regex patterns for running headers/footers to ignore during extraction"
    )


class ChunkingConfig(BaseModel):
    """Configuration options for hybrid semantic-structural chunking."""
    extracted_data_dir: Path = Field(default=EXTRACTED_DATA_DIR)
    chunks_data_dir: Path = Field(default=CHUNKS_DATA_DIR)
    
    target_chunk_tokens: int = Field(default=400, description="Target token budget per chunk")
    max_chunk_tokens: int = Field(default=600, description="Hard upper limit for a single chunk")
    min_chunk_tokens: int = Field(default=80, description="Minimum tokens to avoid micro-chunks")
    chunk_overlap_tokens: int = Field(default=50, description="Token overlap for sliding window prose")
    inject_breadcrumbs: bool = Field(default=True, description="Prefix chunks with document/section breadcrumbs")


class EmbeddingConfig(BaseModel):
    """Configuration for embedding generation."""
    gemini_api_key: str = Field(
        default_factory=lambda: os.getenv("GEMINI_API_KEY", "")
    )
    gemini_embedding_model: str = Field(
        default_factory=lambda: os.getenv("GEMINI_EMBEDDING_MODEL", "models/gemini-embedding-001")
    )
    fallback_local_model: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        description="Local fallback model if Gemini API is offline or rate-limited"
    )
    vector_store_dir: Path = Field(default=VECTOR_STORE_DIR)



class LLMConfig(BaseModel):
    """Configuration for LLM generation with OpenRouter (Qwen3.8 27B & Gemma4 31B) and Gemini fallback."""
    gemini_api_key: str = Field(
        default_factory=lambda: os.getenv("GEMINI_API_KEY", "")
    )
    openrouter_api_key: str = Field(
        default_factory=lambda: os.getenv("OPENROUTER_API_KEY", "")
    )
    openrouter_base_url: str = Field(
        default="https://openrouter.ai/api/v1"
    )
    # Primary model (High-end Gemma-4 31B)
    primary_model: str = Field(
        default_factory=lambda: os.getenv("LLM_PRIMARY_MODEL", "google/gemma-4-31b-it")
    )
    # Complex model for multi-step reasoning, planning & agentic decomposition (Qwen3.8 27B)
    complex_model: str = Field(
        default_factory=lambda: os.getenv("LLM_COMPLEX_MODEL", "qwen/qwen3.8-27b")
    )
    # Fallback model chain in case of quota or rate limit issues
    fallback_models: list[str] = Field(
        default=["qwen/qwen3.8-27b", "google/gemma-4-31b-it", "models/gemini-3.6-flash", "models/gemma-4-31b-it"]
    )
    temperature: float = Field(default=0.1)
    max_output_tokens: int = Field(default=2048)


class RouterConfig(BaseModel):
    """Configuration for Question Router scoring rubric."""
    agentic_threshold: int = Field(
        default=2,
        description="Routing threshold: score >= agentic_threshold triggers Agentic workflow, else Simple RAG"
    )
    # Feature scores per user rubric
    score_direct_factual: int = Field(default=0)
    score_one_focused_retrieval: int = Field(default=0)
    score_requires_comparison: int = Field(default=2)
    score_multiple_independent_facts: int = Field(default=2)
    score_multiple_documents: int = Field(default=2)
    score_query_decomposition: int = Field(default=2)
    score_calculations_or_tools: int = Field(default=3)
    score_iterative_search: int = Field(default=3)
    score_planning_multi_step_reasoning: int = Field(default=3)


class RAGConfig(BaseModel):
    """Configuration for Simple RAG workflow."""
    top_k: int = Field(default=5)
    rrf_k: int = Field(default=60)
    min_relevance_score: float = Field(default=0.015)
    verify_claims: bool = Field(default=True)


class AgentConfig(BaseModel):
    """Configuration for Multi-Step Agentic workflow."""
    max_sub_queries: int = Field(default=4)
    max_rewrite_retries: int = Field(default=2)
    sub_query_top_k: int = Field(default=3)
    verify_claims: bool = Field(default=True)


extraction_config = ExtractionConfig()
chunking_config = ChunkingConfig()
embedding_config = EmbeddingConfig()
llm_config = LLMConfig()
router_config = RouterConfig()
rag_config = RAGConfig()
agent_config = AgentConfig()
