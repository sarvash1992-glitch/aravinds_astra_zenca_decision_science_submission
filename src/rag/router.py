"""Question Router evaluating query complexity using document metadata and exact scoring rubric."""
import logging
from typing import Any, Literal
from pydantic import BaseModel, Field

from src.config import router_config, RouterConfig
from src.llm.client import LLMClient
from src.rag.metadata import get_corpus_context_string

logger = logging.getLogger(__name__)


class FeatureScore(BaseModel):
    """Evaluation of an individual query feature."""
    feature: str = Field(description="Name of the rubric feature")
    present: bool = Field(description="Whether this feature was detected in the query")
    score: int = Field(description="Points contributed to total complexity score")
    rationale: str = Field(default="", description="Brief justification for feature presence")


class RouterDecision(BaseModel):
    """Complete routing decision payload."""
    query: str
    workflow: Literal["simple_rag", "agentic"]
    total_score: int
    threshold: int
    features: list[FeatureScore]
    reasoning: str
    relevant_docs: list[str]
    is_out_of_scope: bool = False
    model_used: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "workflow": self.workflow,
            "total_score": self.total_score,
            "threshold": self.threshold,
            "reasoning": self.reasoning,
            "relevant_docs": self.relevant_docs,
            "is_out_of_scope": self.is_out_of_scope,
            "features": [f.model_dump() for f in self.features],
            "model_used": self.model_used,
        }


class QuestionRouter:
    """Intelligent Question Router classifying queries into Simple RAG or Agentic workflow."""

    def __init__(
        self,
        llm_client: LLMClient | None = None,
        config: RouterConfig | None = None,
    ) -> None:
        self.llm = llm_client or LLMClient()
        self.config = config or router_config
        self.corpus_context = get_corpus_context_string()

    def route(self, query: str) -> RouterDecision:
        """Analyze query complexity and determine appropriate execution workflow.
        
        Args:
            query: The user's input question.
            
        Returns:
            RouterDecision containing workflow selection, score breakdown, and reasoning.
        """
        system_instruction = (
            "You are an expert AI Query Complexity Router for a Retrieval-Augmented Generation system.\n"
            "Your objective is to evaluate incoming questions against our document corpus and score their complexity\n"
            "according to a strict, rigorous rubric."
        )

        prompt = f"""Evaluate the complexity of the following user query for routing between 'simple_rag' and 'agentic' workflows.

{self.corpus_context}

SCORING RUBRIC:
Evaluate each of the following 9 features for the query:
1. "Direct factual question" -> 0 points (True if asking for a direct definition, fact, or concept)
2. "Requires one focused retrieval" -> 0 points (True if single passage search is likely sufficient)
3. "Requires comparison" -> +2 points (True if comparing 2 or more systems, methods, frameworks, or databases)
4. "Requires multiple independent facts" -> +2 points (True if asking for multiple distinct facts across topics)
5. "Requires multiple documents" -> +2 points (True if answering requires information from 2 or more distinct documents in our corpus)
6. "Requires query decomposition" -> +2 points (True if question cannot be answered in 1 shot and must be broken into sub-questions)
7. "Requires calculations or external tools" -> +3 points (True if mathematical calculations, aggregations, or external computation needed)
8. "Requires iterative search" -> +3 points (True if findings from first search inform second search)
9. "Requires planning or multi-step reasoning" -> +3 points (True if synthesizing trade-offs, architectural design, or conditional logic)

Threshold Rule:
- Total Score = Sum of points from all detected features.
- If Total Score >= {self.config.agentic_threshold}: Route to "agentic"
- If Total Score < {self.config.agentic_threshold}: Route to "simple_rag"

USER QUERY:
"{query}"

Return a valid JSON object strictly matching this schema:
{{
  "features": [
    {{"feature": "Direct factual question", "present": true, "score": 0, "rationale": "..."}},
    {{"feature": "Requires one focused retrieval", "present": true, "score": 0, "rationale": "..."}},
    {{"feature": "Requires comparison", "present": false, "score": 0, "rationale": "..."}},
    {{"feature": "Requires multiple independent facts", "present": false, "score": 0, "rationale": "..."}},
    {{"feature": "Requires multiple documents", "present": false, "score": 0, "rationale": "..."}},
    {{"feature": "Requires query decomposition", "present": false, "score": 0, "rationale": "..."}},
    {{"feature": "Requires calculations or external tools", "present": false, "score": 0, "rationale": "..."}},
    {{"feature": "Requires iterative search", "present": false, "score": 0, "rationale": "..."}},
    {{"feature": "Requires planning or multi-step reasoning", "present": false, "score": 0, "rationale": "..."}}
  ],
  "total_score": <int>,
  "reasoning": "<concise explanation of complexity assessment>",
  "relevant_docs": ["<doc1.pdf>", ...],
  "is_out_of_scope": <true if question is completely unrelated to our corpus, false otherwise>
}}
"""
        try:
            # We can use primary model (Gemma) with automatic fallback to Flash
            response_json, model_used = self.llm.generate_json(
                prompt=prompt,
                system_instruction=system_instruction,
                model_tier="primary",
            )
            return self._build_decision(query, response_json, model_used)
        except Exception as e:
            logger.warning("Router LLM failed: %s. Applying robust rule-based fallback routing.", e)
            return self._rule_based_fallback(query)

    def _build_decision(self, query: str, data: dict[str, Any], model_used: str) -> RouterDecision:
        """Parse and validate LLM output into a RouterDecision."""
        raw_features = data.get("features", [])
        features: list[FeatureScore] = []
        calc_total = 0

        rubric_weights = {
            "direct factual question": self.config.score_direct_factual,
            "requires one focused retrieval": self.config.score_one_focused_retrieval,
            "requires comparison": self.config.score_requires_comparison,
            "requires multiple independent facts": self.config.score_multiple_independent_facts,
            "requires multiple documents": self.config.score_multiple_documents,
            "requires query decomposition": self.config.score_query_decomposition,
            "requires calculations or external tools": self.config.score_calculations_or_tools,
            "requires iterative search": self.config.score_iterative_search,
            "requires planning or multi-step reasoning": self.config.score_planning_multi_step_reasoning,
        }

        for item in raw_features:
            name = item.get("feature", "")
            present = bool(item.get("present", False))
            norm_name = name.lower().strip()
            
            # Match weight from rubric
            assigned_weight = 0
            for k, w in rubric_weights.items():
                if k in norm_name or norm_name in k:
                    assigned_weight = w
                    break

            score = assigned_weight if present else 0
            calc_total += score
            features.append(
                FeatureScore(
                    feature=name,
                    present=present,
                    score=score,
                    rationale=item.get("rationale", ""),
                )
            )

        total_score = max(calc_total, int(data.get("total_score", calc_total)))
        workflow: Literal["simple_rag", "agentic"] = (
            "agentic" if total_score >= self.config.agentic_threshold else "simple_rag"
        )

        return RouterDecision(
            query=query,
            workflow=workflow,
            total_score=total_score,
            threshold=self.config.agentic_threshold,
            features=features,
            reasoning=data.get("reasoning", "Evaluated based on scoring rubric."),
            relevant_docs=data.get("relevant_docs", []),
            is_out_of_scope=bool(data.get("is_out_of_scope", False)),
            model_used=model_used,
        )

    def _rule_based_fallback(self, query: str) -> RouterDecision:
        """Deterministic fallback router when LLM API is unavailable."""
        q = query.lower()
        score = 0
        features: list[FeatureScore] = []

        comparison_keywords = ["compare", "versus", "vs", "difference between", "trade-off", "tradeoff", "better than"]
        multi_doc_keywords = ["and", "both", "across all", "compare vector database and rag"]
        planning_keywords = ["how should i design", "how to implement", "production architecture", "step by step", "security considerations"]
        calc_keywords = ["calculate", "formula", "compute", "cost at scale"]

        is_comparison = any(k in q for k in comparison_keywords)
        is_multi_doc = any(k in q for k in multi_doc_keywords) and len(q.split()) > 8
        is_planning = any(k in q for k in planning_keywords)
        is_calc = any(k in q for k in calc_keywords)

        if is_comparison:
            score += self.config.score_requires_comparison
            features.append(FeatureScore(feature="Requires comparison", present=True, score=2, rationale="Keyword match"))
        if is_multi_doc:
            score += self.config.score_multiple_documents
            features.append(FeatureScore(feature="Requires multiple documents", present=True, score=2, rationale="Multi-concept query"))
        if is_planning:
            score += self.config.score_planning_multi_step_reasoning
            features.append(FeatureScore(feature="Requires planning or multi-step reasoning", present=True, score=3, rationale="Complex architectural query"))
        if is_calc:
            score += self.config.score_calculations_or_tools
            features.append(FeatureScore(feature="Requires calculations or external tools", present=True, score=3, rationale="Calculation keywords"))

        workflow: Literal["simple_rag", "agentic"] = (
            "agentic" if score >= self.config.agentic_threshold else "simple_rag"
        )

        return RouterDecision(
            query=query,
            workflow=workflow,
            total_score=score,
            threshold=self.config.agentic_threshold,
            features=features,
            reasoning=f"Rule-based classification assigned score {score}",
            relevant_docs=["rag_architecture_patterns.pdf", "vector_database_comparison.pdf"],
            is_out_of_scope=False,
            model_used="rule_based_fallback",
        )
