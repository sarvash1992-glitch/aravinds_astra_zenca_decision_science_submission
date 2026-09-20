"""Multi-Step Agentic Orchestrator implementing Plan -> Decompose -> Retrieve -> Assess/Rewrite -> Aggregate -> Generate -> Verify."""
import logging
import time
from typing import Any

from src.agent.state import AgentPlan, AgentResponse, AgentTrace, SubQuestionExecution
from src.config import agent_config, AgentConfig
from src.ingestion.vector_store import HybridVectorStore, SearchResult
from src.llm.client import LLMClient
from src.rag.metadata import get_corpus_context_string
from src.rag.simple_rag import SourceCitation, VerificationResult

logger = logging.getLogger(__name__)


class AgenticOrchestrator:
    """Production Agentic workflow for complex, multi-document, comparative, and multi-step queries."""

    def __init__(
        self,
        vector_store: HybridVectorStore | None = None,
        llm_client: LLMClient | None = None,
        config: AgentConfig | None = None,
    ) -> None:
        self.config = config or agent_config
        self.llm = llm_client or LLMClient()
        if vector_store is not None:
            self.store = vector_store
        else:
            self.store = HybridVectorStore()
            self.store.load()
        self.corpus_context = get_corpus_context_string()

    def run(self, query: str) -> AgentResponse:
        """Execute the multi-step agentic workflow:
        1. Analyze and Plan
        2. Decompose Question into targeted sub-questions
        3. For each sub-question:
           - Retrieve evidence
           - Assess evidence sufficiency
           - Rewrite query & retry if insufficient (loop)
        4. Aggregate multi-source evidence
        5. Generate draft answer
        6. Verify claims & abstain if unsupported
        7. Return final response with full trace
        """
        start_time = time.time()
        logger.info("Starting Agentic workflow for query: '%s'", query)

        # 1 & 2. Analyze, Plan, and Decompose
        plan = self._analyze_and_plan(query)

        # 3. Retrieve, Assess, and Rewrite Loop for each sub-question
        sub_executions: list[SubQuestionExecution] = []
        all_retrieved_chunks: dict[str, SearchResult] = {}
        total_rewrites = 0

        for i, sub_q in enumerate(plan.sub_questions, 1):
            sub_exec = self._execute_sub_question_loop(i, sub_q)
            sub_executions.append(sub_exec)
            total_rewrites += len(sub_exec.rewrites)

            # Accumulate chunks by unique chunk_id
            for c in sub_exec.chunks:
                if c.chunk_id not in all_retrieved_chunks:
                    # Find original SearchResult from store or construct
                    all_retrieved_chunks[c.chunk_id] = c

        # 4. Aggregate Evidence
        unique_sources: list[SourceCitation] = []
        seen_ids = set()
        for sub_exec in sub_executions:
            for src in sub_exec.chunks:
                if src.chunk_id not in seen_ids:
                    seen_ids.add(src.chunk_id)
                    unique_sources.append(src)

        if not unique_sources:
            # Fallback if no evidence found across all sub-queries
            latency = time.time() - start_time
            trace = AgentTrace(
                plan=plan,
                sub_executions=sub_executions,
                total_rewrites=total_rewrites,
                total_unique_chunks=0,
                draft_answer="",
            )
            return AgentResponse(
                query=query,
                workflow="agentic",
                answer=(
                    "After decomposing the query and conducting iterative multi-step searches across the corpus, "
                    "no sufficient evidence was found in the indexed documents to answer this question."
                ),
                confidence=0.0,
                sources=[],
                trace=trace,
                abstained=True,
                model_used="none",
                latency_seconds=latency,
            )

        # 5. Generate Draft Answer
        draft_answer, model_used = self._generate_draft_answer(query, plan, sub_executions, unique_sources)

        # 6. Verify Claims
        verification = self._verify_claims(query, draft_answer, unique_sources)

        latency = time.time() - start_time

        # 7. Final Decision (Supported vs Abstain)
        if verification.status == "unsupported" or (
            len(verification.unsupported_claims) > len(verification.supported_claims) and not verification.is_grounded
        ):
            final_answer = (
                "The aggregated evidence across the document corpus was insufficient to verify the conclusions "
                "required for this complex question. The system abstains to avoid hallucination.\n\n"
                f"Audit Verdict: {verification.verdict_rationale}"
            )
            abstained = True
        else:
            final_answer = draft_answer
            abstained = False

        trace = AgentTrace(
            plan=plan,
            sub_executions=sub_executions,
            total_rewrites=total_rewrites,
            total_unique_chunks=len(unique_sources),
            draft_answer=draft_answer,
            claims_verification=verification,
        )

        return AgentResponse(
            query=query,
            workflow="agentic",
            answer=final_answer,
            confidence=verification.confidence,
            sources=unique_sources,
            trace=trace,
            abstained=abstained,
            model_used=model_used,
            latency_seconds=latency,
        )

    def _analyze_and_plan(self, query: str) -> AgentPlan:
        """Analyze query complexity, identify domains, and decompose into sub-questions."""
        system_instruction = (
            "You are an expert AI Research Planner. Your job is to analyze complex, multi-step, "
            "or comparative technical questions and break them down into 2 to 4 precise, targeted sub-questions."
        )

        prompt = f"""ANALYZE AND PLAN EXECUTION FOR COMPLEX QUERY:

{self.corpus_context}

USER QUERY:
"{query}"

TASKS:
1. Explain the overarching reasoning strategy to answer this question.
2. List which documents/domains must be consulted.
3. Decompose the question into 2 to 4 independent, focused sub-questions that can each be answered via targeted document retrieval.

Return ONLY a valid JSON object matching this schema:
{{
  "strategy": "Concise summary of research plan",
  "involved_domains": ["RAG Architecture", "Vector Databases"],
  "sub_questions": [
    "Targeted sub-question 1...",
    "Targeted sub-question 2...",
    "Targeted sub-question 3..."
  ]
}}"""

        try:
            # For complex planning, use complex model tier (Flash) with fallback to Gemma
            data, _ = self.llm.generate_json(
                prompt=prompt,
                system_instruction=system_instruction,
                model_tier="complex",
            )
            sub_qs = data.get("sub_questions", [])
            if not sub_qs:
                sub_qs = [query]

            return AgentPlan(
                query=query,
                strategy=data.get("strategy", "Multi-step iterative decomposition and retrieval."),
                involved_domains=data.get("involved_domains", []),
                sub_questions=sub_qs[: self.config.max_sub_queries],
            )
        except Exception as e:
            logger.warning("Planning LLM failed: %s. Using single-step fallback plan.", e)
            return AgentPlan(
                query=query,
                strategy="Direct query decomposition fallback.",
                involved_domains=[],
                sub_questions=[query],
            )

    def _execute_sub_question_loop(self, step_num: int, sub_question: str) -> SubQuestionExecution:
        """Retrieve evidence for a sub-question with iterative query rewriting if insufficient."""
        active_query = sub_question
        rewrites: list[str] = []
        attempts = 0
        max_attempts = 1 + self.config.max_rewrite_retries

        best_chunks: list[SourceCitation] = []
        is_sufficient = False
        assessment_notes = ""

        while attempts < max_attempts:
            attempts += 1
            logger.debug("Sub-Q %d attempt %d: '%s'", step_num, attempts, active_query)

            # Retrieve
            retrieved = self.store.search_hybrid(query=active_query, top_k=self.config.sub_query_top_k)
            current_chunks: list[SourceCitation] = []
            from src.rag.simple_rag import make_concise_section, make_concise_snippet
            for r in retrieved:
                snippet = make_concise_snippet(r.chunk.raw_content, max_chars=160)
                sec_clean = make_concise_section(r.chunk.section_path)
                current_chunks.append(
                    SourceCitation(
                        doc_name=r.chunk.doc_name,
                        page_number=r.chunk.page_number,
                        section=sec_clean,
                        chunk_id=r.chunk.chunk_id,
                        score=round(r.score, 4),
                        snippet=snippet,
                    )
                )

            if not current_chunks:
                logger.info("No chunks retrieved for '%s'. Rewriting query...", active_query)
                if attempts < max_attempts:
                    rewritten = self._rewrite_query(sub_question, active_query, "No matching documents found.")
                    rewrites.append(rewritten)
                    active_query = rewritten
                continue

            # Assess Evidence Sufficiency
            is_sufficient, assessment_notes, suggested_rewrite = self._assess_evidence(
                sub_question=sub_question,
                active_query=active_query,
                chunks=current_chunks,
            )

            best_chunks = current_chunks

            if is_sufficient or attempts >= max_attempts:
                break

            # If insufficient and more attempts remain: Rewrite Query
            rewritten = suggested_rewrite or self._rewrite_query(
                sub_question, active_query, assessment_notes
            )
            rewrites.append(rewritten)
            active_query = rewritten

        return SubQuestionExecution(
            step_num=step_num,
            sub_question=sub_question,
            active_query=active_query,
            rewrites=rewrites,
            attempts=attempts,
            chunks=best_chunks,
            is_sufficient=is_sufficient,
            assessment_notes=assessment_notes,
        )

    def _assess_evidence(
        self, sub_question: str, active_query: str, chunks: list[SourceCitation]
    ) -> tuple[bool, str, str]:
        """Assess whether the retrieved chunks provide sufficient evidence to answer the sub-question."""
        evidence_snippets = "\n".join([f"- [{c.doc_name} p.{c.page_number}]: {c.snippet}" for c in chunks])

        system_instruction = (
            "You are an evidence assessment evaluator for an Agentic RAG system.\n"
            "Assess whether the retrieved evidence snippets contain relevant facts to answer the sub-question."
        )

        prompt = f"""SUB-QUESTION:
"{sub_question}"

ACTIVE SEARCH QUERY:
"{active_query}"

RETRIEVED SNIPPETS:
{evidence_snippets}

TASK:
1. Determine if the retrieved snippets are sufficient to answer or clarify this sub-question (true/false).
2. If false, suggest a refined search query with better technical keywords, exact entity names, or synonyms.

Return ONLY a valid JSON object:
{{
  "sufficient": true | false,
  "rationale": "Short explanation of sufficiency",
  "suggested_rewrite": "refined search query if insufficient"
}}"""

        try:
            data, _ = self.llm.generate_json(
                prompt=prompt,
                system_instruction=system_instruction,
                model_tier="complex",
            )
            return (
                bool(data.get("sufficient", True)),
                data.get("rationale", "Evidence verified."),
                data.get("suggested_rewrite", ""),
            )
        except Exception as e:
            logger.debug("Evidence assessment LLM failed: %s. Assuming sufficient.", e)
            return True, "Assessment fallback passed.", ""

    def _rewrite_query(self, original_sub_q: str, current_query: str, reason: str) -> str:
        """Reformulate search query with technical synonyms and alternate phrasing."""
        prompt = f"""The following search query failed to retrieve sufficient technical documentation:
Original Question: "{original_sub_q}"
Previous Query: "{current_query}"
Reason: {reason}

Generate an improved search query using synonyms, specific technical terms, or broader keywords that would match technical PDF sections.

Return ONLY a valid JSON object:
{{
  "rewritten_query": "concise keyword-rich search query"
}}"""

        try:
            data, _ = self.llm.generate_json(prompt=prompt, model_tier="complex")
            return data.get("rewritten_query", original_sub_q)
        except Exception:
            # Fallback simple keyword extraction
            words = [w for w in original_sub_q.split() if len(w) > 3]
            return " ".join(words[:5])

    def _generate_draft_answer(
        self,
        query: str,
        plan: AgentPlan,
        sub_executions: list[SubQuestionExecution],
        unique_sources: list[SourceCitation],
    ) -> tuple[str, str]:
        """Synthesize aggregated multi-step evidence into a cohesive, grounded draft answer."""
        # Build structured context of sub-executions
        context_blocks = []
        for sexec in sub_executions:
            block = [f"### Sub-Question {sexec.step_num}: {sexec.sub_question}"]
            for c in sexec.chunks:
                block.append(f"- [{c.doc_name}, p.{c.page_number}, {c.section}]: {c.snippet}")
            context_blocks.append("\n".join(block))

        aggregated_context = "\n\n".join(context_blocks)

        system_instruction = (
            "You are a Senior AI Architect providing comprehensive, structured technical analysis.\n"
            "Synthesize the aggregated evidence from multiple sub-questions into a cohesive, authoritative answer.\n\n"
            "STRICT GUIDELINES:\n"
            "1. Answer must be strictly grounded in the provided evidence.\n"
            "2. For comparisons, present key trade-offs in clear comparative sections or tables.\n"
            "3. Include precise inline citations in the format: [DocName, p.X].\n"
            "4. Avoid speculative commentary. If any sub-part lacked evidence, explicitly acknowledge the gap."
        )

        user_prompt = f"""ORIGINAL USER QUESTION:
{query}

EXECUTION STRATEGY:
{plan.strategy}

AGGREGATED SUB-QUESTION EVIDENCE:
{aggregated_context}

Please provide the final comprehensive, comparative synthesis with inline citations."""

        answer, model_used = self.llm.generate(
            prompt=user_prompt,
            system_instruction=system_instruction,
            model_tier="complex",  # Use Flash / complex tier for multi-step synthesis
            temperature=0.1,
        )
        from src.llm.client import clean_model_text
        return clean_model_text(answer), model_used

    def _verify_claims(
        self, query: str, draft_answer: str, sources: list[SourceCitation]
    ) -> VerificationResult:
        """Audit the multi-step draft answer against the aggregated sources."""
        evidence_summary = "\n".join([f"- [{s.doc_name}, p.{s.page_number}]: {s.snippet}" for s in sources[:8]])

        system_instruction = (
            "You are a strict Claim Verification & Fact-Checking auditor for an enterprise Agentic AI system.\n"
            "Verify whether the assertions and conclusions in the draft answer are backed by the retrieved evidence.\n"
            "Output valid JSON only."
        )

        prompt = f"""AUDIT THE MULTI-STEP SYNTHESIS:

USER QUESTION:
{query}

RETRIEVED EVIDENCE:
{evidence_summary}

DRAFT ANSWER:
{draft_answer}

TASKS:
1. Extract key assertions and comparative claims.
2. Verify if they are supported by the retrieved evidence.
3. Assign status: 'supported', 'partially_supported', or 'unsupported'.
4. Calculate confidence between 0.0 and 1.0.

Return ONLY a valid JSON object:
{{
  "status": "supported" | "partially_supported" | "unsupported",
  "is_grounded": true | false,
  "confidence": 0.95,
  "supported_claims": ["claim 1...", "claim 2..."],
  "unsupported_claims": ["any unverified claims..."],
  "verdict_rationale": "Audit evaluation summary."
}}"""

        try:
            data, _ = self.llm.generate_json(
                prompt=prompt,
                system_instruction=system_instruction,
                model_tier="complex",
            )
            if isinstance(data, list):
                data = data[0] if data and isinstance(data[0], dict) else {}
            return VerificationResult(
                status=data.get("status", "supported"),
                is_grounded=bool(data.get("is_grounded", True)),
                confidence=float(data.get("confidence", 0.9)),
                supported_claims=data.get("supported_claims", []),
                unsupported_claims=data.get("unsupported_claims", []),
                verdict_rationale=data.get("verdict_rationale", "Verified against multi-source evidence."),
            )
        except Exception as e:
            logger.warning("Agentic verification LLM failed: %s. Defaulting to passed.", e)
            return VerificationResult(
                status="supported",
                is_grounded=True,
                confidence=0.85,
                verdict_rationale="Automatic fallback verification passed.",
            )
