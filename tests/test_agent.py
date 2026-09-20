"""Unit tests for the Agentic Orchestrator and state models."""
import pytest
from src.agent.orchestrator import AgenticOrchestrator
from src.agent.state import AgentPlan, AgentResponse, AgentTrace, SubQuestionExecution
from src.rag.simple_rag import SourceCitation, VerificationResult


def test_agent_plan_creation():
    """Verify AgentPlan structure."""
    plan = AgentPlan(
        query="Compare FAISS and Pinecone",
        strategy="Decompose into FAISS specs and Pinecone specs, then compare.",
        involved_domains=["Vector Databases"],
        sub_questions=["What is FAISS latency?", "What is Pinecone cost?"],
    )
    assert len(plan.sub_questions) == 2
    assert "Vector Databases" in plan.involved_domains


def test_sub_question_execution_record():
    """Verify SubQuestionExecution tracking."""
    step = SubQuestionExecution(
        step_num=1,
        sub_question="What is FAISS p50 latency?",
        active_query="FAISS p50 latency 1M vectors",
        rewrites=["FAISS p50 latency 1M vectors"],
        attempts=2,
        chunks=[],
        is_sufficient=True,
        assessment_notes="Found benchmarks table",
    )
    assert step.step_num == 1
    assert step.attempts == 2
    assert len(step.rewrites) == 1
    assert step.is_sufficient is True


def test_agent_trace_serialization():
    """Verify full trace dictionary serialization."""
    plan = AgentPlan(
        query="test query",
        strategy="test strategy",
        sub_questions=["sub1"],
    )
    step = SubQuestionExecution(
        step_num=1,
        sub_question="sub1",
        active_query="sub1",
    )
    trace = AgentTrace(
        plan=plan,
        sub_executions=[step],
        total_rewrites=0,
        total_unique_chunks=0,
        draft_answer="Test draft",
    )
    res = AgentResponse(
        query="test query",
        answer="Final answer",
        confidence=0.9,
        sources=[],
        trace=trace,
        abstained=False,
    )
    data = res.to_dict()
    assert data["workflow"] == "agentic"
    assert data["answer"] == "Final answer"
    assert "plan" in data["trace"]
    assert len(data["trace"]["sub_executions"]) == 1
