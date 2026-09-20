"""Integration and contract tests for FastAPI endpoints."""
import pytest
from fastapi.testclient import TestClient
from src.api.main import app

client = TestClient(app)


def test_api_root():
    """Verify root status endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "Intelligent Knowledge Assistant" in data["message"]
    assert data["docs"] == "/docs"


def test_api_health():
    """Verify health check returns corpus and model metadata."""
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["total_indexed_chunks"] > 0
    assert len(data["corpus_documents"]) == 3
    assert "rag_architecture_patterns.pdf" in data["corpus_documents"]


def test_api_metadata():
    """Verify corpus metadata endpoint."""
    response = client.get("/api/v1/metadata")
    assert response.status_code == 200
    data = response.json()
    assert "vector_database_comparison.pdf" in data
    assert "agentic_ai_frameworks.pdf" in data
    assert len(data["rag_architecture_patterns.pdf"]["topics"]) > 0


def test_api_route_endpoint():
    """Verify router scoring endpoint without running full generation."""
    response = client.post(
        "/api/v1/route",
        json={"query": "What is Reciprocal Rank Fusion?"},
    )
    assert response.status_code == 200
    data = response.json()
    assert "workflow" in data
    assert "total_score" in data
    assert "features" in data
    assert isinstance(data["features"], list)


def test_api_query_simple_rag_contract():
    """Verify full query endpoint request/response contract for simple RAG."""
    response = client.post(
        "/api/v1/query",
        json={"query": "What is Reciprocal Rank Fusion?", "force_workflow": "simple_rag"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["workflow"] == "simple_rag"
    assert "answer" in data
    assert len(data["answer"]) > 0
    assert "sources" in data
    assert "verification" in data
    assert "latency_seconds" in data
