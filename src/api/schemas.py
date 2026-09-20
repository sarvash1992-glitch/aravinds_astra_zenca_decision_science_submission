"""API request and response schemas matching the Technical Assessment contract."""
from typing import Any, Literal
from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    """User question payload."""
    query: str = Field(..., min_length=2, description="The technical question to answer", example="What is Reciprocal Rank Fusion?")
    force_workflow: Literal["auto", "simple_rag", "agentic"] = Field(
        default="auto",
        description="Routing policy: 'auto' (use router score), 'simple_rag', or 'agentic'",
    )
    top_k: int | None = Field(default=None, ge=1, le=20, description="Max chunks to retrieve")


class SourceCitationSchema(BaseModel):
    """Citation details for an answer source."""
    doc_name: str
    page_number: int
    section: str
    chunk_id: str
    score: float
    snippet: str


class FeatureScoreSchema(BaseModel):
    """Rubric evaluation score for an individual feature."""
    feature: str
    present: bool
    score: int
    rationale: str


class RouterDecisionSchema(BaseModel):
    """Breakdown of question complexity scoring and workflow selection."""
    query: str
    workflow: Literal["simple_rag", "agentic"]
    total_score: int
    threshold: int
    features: list[FeatureScoreSchema]
    reasoning: str
    relevant_docs: list[str]
    is_out_of_scope: bool
    model_used: str


class VerificationResultSchema(BaseModel):
    """Factuality and claim verification audit result."""
    status: Literal["supported", "unsupported", "partially_supported"]
    is_grounded: bool
    confidence: float
    supported_claims: list[str]
    unsupported_claims: list[str]
    verdict_rationale: str


class SubQuestionStepSchema(BaseModel):
    """Agentic execution trace for a single decomposed sub-question."""
    step_num: int
    sub_question: str
    active_query: str
    rewrites: list[str]
    attempts: int
    chunks: list[SourceCitationSchema]
    is_sufficient: bool
    assessment_notes: str


class AgentPlanSchema(BaseModel):
    """Agentic strategy and decomposition plan."""
    query: str
    strategy: str
    involved_domains: list[str]
    sub_questions: list[str]


class AgentTraceSchema(BaseModel):
    """Trace of all agent steps for multi-step queries."""
    plan: AgentPlanSchema
    sub_executions: list[SubQuestionStepSchema]
    total_rewrites: int
    total_unique_chunks: int
    draft_answer: str
    claims_verification: VerificationResultSchema | None = None


class QueryResponse(BaseModel):
    """Standardized production response contract."""
    query: str
    workflow: Literal["simple_rag", "agentic"]
    answer: str
    confidence: float
    sources: list[SourceCitationSchema]
    routing: RouterDecisionSchema
    verification: VerificationResultSchema
    agent_trace: AgentTraceSchema | None = None
    abstained: bool
    model_used: str
    latency_seconds: float


class HealthResponse(BaseModel):
    """Service health and index statistics."""
    status: str
    total_indexed_chunks: int
    faiss_dimension: int
    primary_model: str
    complex_model: str
    corpus_documents: list[str]
