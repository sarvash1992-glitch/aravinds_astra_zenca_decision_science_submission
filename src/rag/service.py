"""Unified Knowledge Assistant Service routing queries to Simple RAG or Agentic workflow."""
import logging
import time
from typing import Any, Literal
from pydantic import BaseModel, Field

from src.agent.orchestrator import AgenticOrchestrator
from src.agent.state import AgentTrace
from src.config import router_config
from src.ingestion.vector_store import HybridVectorStore
from src.llm.client import LLMClient
from src.rag.router import QuestionRouter, RouterDecision
from src.rag.simple_rag import SimpleRAGPipeline, SourceCitation, VerificationResult

logger = logging.getLogger(__name__)


class QueryResult(BaseModel):
    """Unified result payload for any question answering request."""
    query: str
    workflow: Literal["simple_rag", "agentic"]
    answer: str
    confidence: float
    sources: list[SourceCitation]
    routing: RouterDecision
    verification: VerificationResult
    agent_trace: AgentTrace | None = None
    abstained: bool = False
    model_used: str = ""
    latency_seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "workflow": self.workflow,
            "answer": self.answer,
            "confidence": round(self.confidence, 4),
            "sources": [s.model_dump() for s in self.sources],
            "routing": self.routing.to_dict(),
            "verification": self.verification.model_dump(),
            "agent_trace": self.agent_trace.model_dump() if self.agent_trace else None,
            "abstained": self.abstained,
            "model_used": self.model_used,
            "latency_seconds": round(self.latency_seconds, 3),
        }


class KnowledgeAssistantService:
    """Unified service orchestrating router evaluation, simple RAG, and agentic workflows."""

    def __init__(
        self,
        vector_store: HybridVectorStore | None = None,
        llm_client: LLMClient | None = None,
    ) -> None:
        self.llm = llm_client or LLMClient()
        self.store = vector_store or HybridVectorStore()
        if self.store.faiss_index is None:
            self.store.load()

        self.router = QuestionRouter(llm_client=self.llm)
        self.simple_rag = SimpleRAGPipeline(vector_store=self.store, llm_client=self.llm)
        self.agentic = AgenticOrchestrator(vector_store=self.store, llm_client=self.llm)

    def answer_query(
        self,
        query: str,
        force_workflow: Literal["auto", "simple_rag", "agentic"] = "auto",
        top_k: int | None = None,
    ) -> QueryResult:
        """Process question by routing to appropriate workflow or adhering to manual override."""
        start_time = time.time()
        logger.info("Processing query: '%s' (force_workflow: %s)", query, force_workflow)

        # 1. Evaluate Question Complexity via Router
        routing_decision = self.router.route(query)

        # Determine effective workflow
        if force_workflow in ("simple_rag", "agentic"):
            effective_workflow = force_workflow
            logger.info("Workflow forced to: %s", effective_workflow)
        else:
            effective_workflow = routing_decision.workflow
            logger.info(
                "Router selected workflow: %s (score: %d, threshold: %d)",
                effective_workflow,
                routing_decision.total_score,
                routing_decision.threshold,
            )

        # 2. Execute selected workflow
        if effective_workflow == "agentic":
            agent_res = self.agentic.run(query)
            latency = time.time() - start_time
            verification = agent_res.trace.claims_verification or VerificationResult(
                status="supported" if not agent_res.abstained else "unsupported",
                is_grounded=not agent_res.abstained,
                confidence=agent_res.confidence,
                verdict_rationale="Agentic verification passed.",
            )
            return QueryResult(
                query=query,
                workflow="agentic",
                answer=agent_res.answer,
                confidence=agent_res.confidence,
                sources=agent_res.sources,
                routing=routing_decision,
                verification=verification,
                agent_trace=agent_res.trace,
                abstained=agent_res.abstained,
                model_used=agent_res.model_used,
                latency_seconds=latency,
            )
        else:
            rag_res = self.simple_rag.run(query, top_k=top_k)
            latency = time.time() - start_time
            return QueryResult(
                query=query,
                workflow="simple_rag",
                answer=rag_res.answer,
                confidence=rag_res.confidence,
                sources=rag_res.sources,
                routing=routing_decision,
                verification=rag_res.verification,
                agent_trace=None,
                abstained=rag_res.abstained,
                model_used=rag_res.model_used,
                latency_seconds=latency,
            )
