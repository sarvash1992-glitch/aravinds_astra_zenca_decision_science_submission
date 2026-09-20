"""Unit tests for the QuestionRouter and scoring rubric."""
import pytest
from src.config import RouterConfig
from src.rag.router import QuestionRouter, RouterDecision, FeatureScore


class MockLLMClient:
    """Mock LLM client returning deterministic JSON for testing."""
    def __init__(self, mock_response: dict):
        self.mock_response = mock_response

    def generate_json(self, *args, **kwargs):
        return self.mock_response, "mock-model"


def test_router_factual_query_scores_zero():
    """Direct factual question should score 0 and route to simple_rag."""
    mock_data = {
        "features": [
            {"feature": "Direct factual question", "present": True, "score": 0, "rationale": "Asks definition"},
            {"feature": "Requires one focused retrieval", "present": True, "score": 0, "rationale": "Single chunk"},
            {"feature": "Requires comparison", "present": False, "score": 0, "rationale": ""},
            {"feature": "Requires multiple independent facts", "present": False, "score": 0, "rationale": ""},
            {"feature": "Requires multiple documents", "present": False, "score": 0, "rationale": ""},
            {"feature": "Requires query decomposition", "present": False, "score": 0, "rationale": ""},
            {"feature": "Requires calculations or external tools", "present": False, "score": 0, "rationale": ""},
            {"feature": "Requires iterative search", "present": False, "score": 0, "rationale": ""},
            {"feature": "Requires planning or multi-step reasoning", "present": False, "score": 0, "rationale": ""},
        ],
        "total_score": 0,
        "reasoning": "Simple factual question.",
        "relevant_docs": ["rag_architecture_patterns.pdf"],
        "is_out_of_scope": False,
    }
    router = QuestionRouter(llm_client=MockLLMClient(mock_data))
    decision = router.route("What is Reciprocal Rank Fusion?")

    assert decision.workflow == "simple_rag"
    assert decision.total_score == 0
    assert len(decision.features) == 9
    assert decision.is_out_of_scope is False


def test_router_comparison_query_routes_to_agentic():
    """Comparison query should receive +2 points and route to agentic (threshold = 2)."""
    mock_data = {
        "features": [
            {"feature": "Requires comparison", "present": True, "score": 2, "rationale": "Compares FAISS and Pinecone"},
            {"feature": "Requires multiple independent facts", "present": True, "score": 2, "rationale": "Latency and cost"},
        ],
        "total_score": 4,
        "reasoning": "Requires cross-database comparative analysis.",
        "relevant_docs": ["vector_database_comparison.pdf"],
        "is_out_of_scope": False,
    }
    router = QuestionRouter(llm_client=MockLLMClient(mock_data))
    decision = router.route("Compare FAISS and Pinecone latency and cost trade-offs")

    assert decision.workflow == "agentic"
    assert decision.total_score == 4
    assert decision.threshold == 2


def test_router_rule_based_fallback():
    """Router should fallback gracefully if LLM raises exception."""
    class FailingLLMClient:
        def generate_json(self, *args, **kwargs):
            raise ConnectionError("API Unavailable")

    router = QuestionRouter(llm_client=FailingLLMClient())
    decision = router.route("Compare Pinecone vs Weaviate")

    assert decision.workflow == "agentic"
    assert decision.total_score >= 2
    assert decision.model_used == "rule_based_fallback"
