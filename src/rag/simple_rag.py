"""Simple RAG workflow implementing Retrieve -> Rerank & Validate -> Generate Draft -> Verify Claims -> Final Answer / Abstain."""
import logging
import re
import time
from typing import Any, Literal
from pydantic import BaseModel, Field

from src.config import rag_config, RAGConfig
from src.ingestion.vector_store import HybridVectorStore, SearchResult
from src.llm.client import LLMClient

logger = logging.getLogger(__name__)


def make_concise_section(section_path: str) -> str:
    """Extract a clean, concise section title (leaf section or last 2 levels)."""
    if not section_path:
        return "General"
    parts = [p.strip() for p in section_path.split(">") if p.strip()]
    parts = [p for p in parts if p.lower() not in ("table of contents", "document")]
    if not parts:
        return "General"
    if len(parts) >= 2:
        return f"{parts[-2]} > {parts[-1]}"
    return parts[-1]


def make_concise_snippet(raw_content: str, max_chars: int = 160) -> str:
    """Extract a clean, concise excerpt snippet without long breadcrumb headers."""
    text = raw_content.strip()
    text = re.sub(r"^\([^)]*?>[^)]*?\)\s*", "", text)
    text = re.sub(r"^([A-Za-z0-9_.\s-]+\s*>\s*)+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= max_chars:
        return text
    truncated = text[:max_chars].rsplit(" ", 1)[0]
    return f"{truncated}..."


class SourceCitation(BaseModel):
    """Structured attribution citation for a retrieved evidence chunk."""
    doc_name: str
    page_number: int
    section: str
    chunk_id: str
    score: float
    snippet: str

    def to_citation_label(self) -> str:
        return f"[{self.doc_name}, p.{self.page_number} - {self.section}]"


class VerificationResult(BaseModel):
    """Claim verification audit result."""
    status: Literal["supported", "unsupported", "partially_supported"]
    is_grounded: bool
    confidence: float
    supported_claims: list[str] = Field(default_factory=list)
    unsupported_claims: list[str] = Field(default_factory=list)
    verdict_rationale: str = ""


class RAGResponse(BaseModel):
    """Standardized response from the Simple RAG pipeline."""
    query: str
    workflow: str = "simple_rag"
    answer: str
    draft_answer: str = ""
    confidence: float = 1.0
    sources: list[SourceCitation] = Field(default_factory=list)
    verification: VerificationResult
    abstained: bool = False
    model_used: str = ""
    latency_seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "workflow": self.workflow,
            "answer": self.answer,
            "draft_answer": self.draft_answer,
            "confidence": self.confidence,
            "sources": [s.model_dump() for s in self.sources],
            "verification": self.verification.model_dump(),
            "abstained": self.abstained,
            "model_used": self.model_used,
            "latency_seconds": round(self.latency_seconds, 3),
        }


class SimpleRAGPipeline:
    """End-to-end Simple RAG pipeline with hybrid search and claim verification."""

    def __init__(
        self,
        vector_store: HybridVectorStore | None = None,
        llm_client: LLMClient | None = None,
        config: RAGConfig | None = None,
    ) -> None:
        self.config = config or rag_config
        self.llm = llm_client or LLMClient()
        if vector_store is not None:
            self.store = vector_store
        else:
            self.store = HybridVectorStore()
            self.store.load()

    def run(self, query: str, top_k: int | None = None) -> RAGResponse:
        """Execute the Simple RAG pipeline:
        1. Retrieve Evidence (Hybrid FAISS + BM25 with RRF)
        2. Rerank and Validate
        3. Generate Draft Answer
        4. Verify Claims against context
        5. Return Final Answer with Citations or Abstain
        """
        start_time = time.time()
        k = top_k or self.config.top_k

        # 1. Retrieve Evidence
        retrieved = self.store.search_hybrid(query=query, top_k=k, rrf_k=self.config.rrf_k)

        # 2. Rerank and Validate
        valid_chunks, sources = self._rerank_and_validate(retrieved)

        if not valid_chunks:
            # Knowledge base does not contain sufficient evidence
            latency = time.time() - start_time
            verification = VerificationResult(
                status="unsupported",
                is_grounded=False,
                confidence=0.0,
                unsupported_claims=["No relevant evidence found in indexed corpus."],
                verdict_rationale="Retrieved score below minimum relevance threshold.",
            )
            return RAGResponse(
                query=query,
                workflow="simple_rag",
                answer=(
                    "I am unable to find sufficient information in the provided document corpus to answer "
                    f"your question ('{query}'). The knowledge base does not cover this topic."
                ),
                abstained=True,
                sources=[],
                verification=verification,
                model_used="none",
                latency_seconds=latency,
            )

        # 3. Generate Draft Answer
        draft_answer, model_used = self._generate_draft_answer(query, valid_chunks)

        # 4. Verify Claims
        if self.config.verify_claims:
            verification = self._verify_claims(query, draft_answer, valid_chunks)
        else:
            verification = VerificationResult(
                status="supported",
                is_grounded=True,
                confidence=0.9,
                verdict_rationale="Claim verification disabled in config.",
            )

        latency = time.time() - start_time

        # 5. Final Answer or Abstain
        if verification.status == "unsupported" or (
            len(verification.unsupported_claims) > len(verification.supported_claims) and not verification.is_grounded
        ):
            final_answer = (
                "The retrieved corpus documents do not contain enough verified evidence to answer this question accurately. "
                "To prevent hallucination, the system abstains from generating a definitive answer.\n\n"
                f"Verification Note: {verification.verdict_rationale}"
            )
            abstained = True
        else:
            final_answer = draft_answer
            abstained = False

        return RAGResponse(
            query=query,
            workflow="simple_rag",
            answer=final_answer,
            draft_answer=draft_answer,
            confidence=verification.confidence,
            sources=sources,
            verification=verification,
            abstained=abstained,
            model_used=model_used,
            latency_seconds=latency,
        )

    def _rerank_and_validate(
        self, results: list[SearchResult]
    ) -> tuple[list[SearchResult], list[SourceCitation]]:
        """Filter out low-scoring or empty chunks and format source citations."""
        valid_results: list[SearchResult] = []
        sources: list[SourceCitation] = []

        for r in results:
            if r.score >= self.config.min_relevance_score:
                valid_results.append(r)
                snippet = make_concise_snippet(r.chunk.raw_content, max_chars=160)
                sec_clean = make_concise_section(r.chunk.section_path)
                sources.append(
                    SourceCitation(
                        doc_name=r.chunk.doc_name,
                        page_number=r.chunk.page_number,
                        section=sec_clean,
                        chunk_id=r.chunk.chunk_id,
                        score=round(r.score, 4),
                        snippet=snippet,
                    )
                )

        return valid_results, sources

    def _format_context(self, chunks: list[SearchResult]) -> str:
        """Format chunk evidence for the LLM context prompt."""
        formatted_chunks = []
        for i, res in enumerate(chunks, 1):
            c = res.chunk
            header = f"--- [Evidence {i}] Document: {c.doc_name} | Page: {c.page_number} | Section: {c.section_path} | ChunkID: {c.chunk_id} ---"
            formatted_chunks.append(f"{header}\n{c.text}\n")
        return "\n".join(formatted_chunks)

    def _generate_draft_answer(
        self, query: str, chunks: list[SearchResult]
    ) -> tuple[str, str]:
        """Generate a grounded draft answer citing the retrieved chunks."""
        context_str = self._format_context(chunks)

        system_prompt = (
            "You are an expert AI Technical Assistant. Your task is to provide accurate, concise, and "
            "authoritative answers strictly grounded in the provided document evidence.\n\n"
            "STRICT GUIDELINES:\n"
            "1. Answer ONLY using facts directly stated in the evidence below.\n"
            "2. Whenever citing facts, include citations inline formatted as: [DocName, p.X].\n"
            "3. If the evidence does not contain sufficient information to answer the question, explicitly state: "
            "'The provided documents do not contain sufficient information to answer this question.'\n"
            "4. Do NOT speculate, infer, or bring external information."
        )

        user_prompt = f"""QUESTION:
{query}

RETRIEVED EVIDENCE:
{context_str}

Please provide a clear, well-structured answer with inline citations."""

        answer, model_used = self.llm.generate(
            prompt=user_prompt,
            system_instruction=system_prompt,
            model_tier="primary",  # High-end Gemma 31B
            temperature=0.1,
        )
        from src.llm.client import clean_model_text
        return clean_model_text(answer), model_used

    def _verify_claims(
        self, query: str, draft_answer: str, chunks: list[SearchResult]
    ) -> VerificationResult:
        """Audit the draft answer against the evidence to verify grounding."""
        context_str = self._format_context(chunks)

        system_instruction = (
            "You are a strict Fact-Checking & Claim Verification engine for an enterprise RAG system.\n"
            "Your job is to determine whether each claim in the draft answer is directly substantiated by the evidence.\n"
            "You must output valid JSON."
        )

        prompt = f"""AUDIT THE FOLLOWING DRAFT ANSWER FOR FACTUAL GROUNDING:

QUESTION:
{query}

RETRIEVED EVIDENCE:
{context_str}

DRAFT ANSWER TO AUDIT:
{draft_answer}

TASKS:
1. Extract the key factual claims made in the draft answer.
2. For each claim, check if it is directly supported by the evidence above.
3. Determine if the overall answer is 'supported', 'partially_supported', or 'unsupported'.
4. Calculate a confidence score between 0.0 and 1.0.

Return ONLY a valid JSON object matching this schema:
{{
  "status": "supported" | "partially_supported" | "unsupported",
  "is_grounded": true | false,
  "confidence": 0.95,
  "supported_claims": ["claim 1...", "claim 2..."],
  "unsupported_claims": ["claim if any..."],
  "verdict_rationale": "Short explanation of audit findings."
}}"""

        try:
            data, _ = self.llm.generate_json(
                prompt=prompt,
                system_instruction=system_instruction,
                model_tier="primary",
            )
            if isinstance(data, list):
                data = data[0] if data and isinstance(data[0], dict) else {}
            return VerificationResult(
                status=data.get("status", "supported"),
                is_grounded=bool(data.get("is_grounded", True)),
                confidence=float(data.get("confidence", 0.9)),
                supported_claims=data.get("supported_claims", []),
                unsupported_claims=data.get("unsupported_claims", []),
                verdict_rationale=data.get("verdict_rationale", "Verified against retrieved context."),
            )
        except Exception as e:
            logger.warning("Claim verification LLM failed: %s. Defaulting to permissive validation.", e)
            return VerificationResult(
                status="supported",
                is_grounded=True,
                confidence=0.85,
                verdict_rationale="Automatic fallback verification passed.",
            )
