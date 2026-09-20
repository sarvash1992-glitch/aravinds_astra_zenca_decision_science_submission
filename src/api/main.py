"""FastAPI application providing production REST endpoints for the Knowledge Assistant."""
import logging
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Query, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.api.schemas import (
    HealthResponse,
    QueryRequest,
    QueryResponse,
    RouterDecisionSchema,
)
from src.config import llm_config
from src.rag.metadata import CORPUS_METADATA
from src.rag.service import KnowledgeAssistantService

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# Service singleton
assistant_service: KnowledgeAssistantService | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize resources on startup and clean up on shutdown."""
    global assistant_service
    logger.info("Initializing Knowledge Assistant Service...")
    assistant_service = KnowledgeAssistantService()
    logger.info("Service initialized with %d indexed chunks.", len(assistant_service.store.chunks))
    yield
    logger.info("Shutting down Knowledge Assistant Service...")


app = FastAPI(
    title="Intelligent Knowledge Assistant API",
    description=(
        "Production-grade Knowledge Assistant combining Simple RAG and Multi-Step Agentic workflows. "
        "Evaluates query complexity via a 9-feature scoring rubric, employs FAISS + BM25 Reciprocal Rank Fusion, "
        "and executes fact-checking claim verification to eliminate hallucinations."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_service() -> KnowledgeAssistantService:
    global assistant_service
    if assistant_service is None:
        assistant_service = KnowledgeAssistantService()
    return assistant_service


@app.get("/", tags=["System"])
async def root() -> dict[str, str]:
    """Root endpoint welcoming users and pointing to documentation."""
    return {
        "message": "Intelligent Knowledge Assistant API is running.",
        "docs": "/docs",
        "openapi": "/openapi.json",
    }


@app.get("/api/v1/health", response_model=HealthResponse, tags=["System"])
async def health_check() -> HealthResponse:
    """Check system health, vector store statistics, and LLM configuration."""
    service = get_service()
    dim = service.store.faiss_index.d if service.store.faiss_index else 0
    return HealthResponse(
        status="healthy",
        total_indexed_chunks=len(service.store.chunks),
        faiss_dimension=dim,
        primary_model=llm_config.primary_model,
        complex_model=llm_config.complex_model,
        corpus_documents=list(CORPUS_METADATA.keys()),
    )


@app.get("/api/v1/metadata", tags=["Knowledge Base"])
async def get_corpus_metadata() -> dict[str, Any]:
    """Retrieve metadata catalog for all indexed corpus documents."""
    return {
        doc_name: {
            "title": meta.title,
            "domain": meta.domain,
            "topics": meta.topics,
            "entities": meta.entities,
            "summary": meta.summary,
            "sample_queries": meta.suitable_queries,
        }
        for doc_name, meta in CORPUS_METADATA.items()
    }


@app.post("/api/v1/route", response_model=RouterDecisionSchema, tags=["Routing"])
async def evaluate_route(request: QueryRequest) -> RouterDecisionSchema:
    """Evaluate query complexity against rubric scoring without running full generation."""
    service = get_service()
    decision = service.router.route(request.query)
    return RouterDecisionSchema.model_validate(decision.to_dict())


@app.post("/api/v1/query", response_model=QueryResponse, tags=["Question Answering"])
async def answer_query(request: QueryRequest) -> QueryResponse:
    """Answer technical questions over the knowledge base with automatic workflow routing.
    
    - Routes to Simple RAG or Agentic based on the 9-feature complexity rubric.
    - Grounded responses with precise document and page citations.
    - Automated claim verification to detect and prevent hallucination.
    """
    service = get_service()
    try:
        result = service.answer_query(
            query=request.query,
            force_workflow=request.force_workflow,
            top_k=request.top_k,
        )
        return QueryResponse.model_validate(result.to_dict())
    except Exception as e:
        logger.error("Error processing query '%s': %s", request.query, e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process query: {str(e)}",
        )
