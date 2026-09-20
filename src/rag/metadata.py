"""Metadata catalog of the documents in the knowledge base."""
from dataclasses import dataclass, field
from typing import Any


@dataclass
class DocumentMetadata:
    """Metadata describing a document's domain, coverage, and contents."""
    doc_name: str
    title: str
    domain: str
    topics: list[str]
    entities: list[str]
    summary: str
    suitable_queries: list[str] = field(default_factory=list)


CORPUS_METADATA: dict[str, DocumentMetadata] = {
    "rag_architecture_patterns.pdf": DocumentMetadata(
        doc_name="rag_architecture_patterns.pdf",
        title="RAG Architecture Patterns & Production Implementation",
        domain="RAG Architecture & Information Retrieval",
        topics=[
            "Document chunking strategies (fixed, semantic, structural)",
            "Dense vector search vs sparse BM25 retrieval",
            "Hybrid search and Reciprocal Rank Fusion (RRF)",
            "Cross-encoder re-ranking and context compression",
            "Query expansion, rewriting, and HyDE",
            "Evaluation metrics (hit rate, MRR, NDCG, faithfulness, answer relevance)",
        ],
        entities=[
            "FAISS", "BM25", "Reciprocal Rank Fusion (RRF)", "Bi-encoders", "Cross-encoders",
            "Sentence Transformers", "HyDE", "Ragas", "TruLens",
        ],
        summary=(
            "Comprehensive technical guide detailing architectural patterns for production RAG systems, "
            "evaluating trade-offs between dense semantic retrieval and sparse lexical search, "
            "re-ranking mechanisms, chunking granularities, and evaluation methodologies."
        ),
        suitable_queries=[
            "What is Reciprocal Rank Fusion?",
            "How does hybrid search combine dense and sparse retrievals?",
            "What are common chunking strategies for technical PDFs?",
        ],
    ),
    "vector_database_comparison.pdf": DocumentMetadata(
        doc_name="vector_database_comparison.pdf",
        title="Comparative Analysis of Vector Databases",
        domain="Vector Storage & Database Benchmarking",
        topics=[
            "Comparative analysis of FAISS, Pinecone, Weaviate, pgvector, Chroma",
            "Query latency (p50, p99) at 1M and 10M vector scale",
            "Index structures: HNSW, IVF-PQ, Flat indexing",
            "Metadata filtering strategies (pre-filtering, post-filtering, single-stage)",
            "Operational complexity: embedded vs cloud-managed vs self-hosted",
            "Cost models, pricing tiers, and hardware resource utilization",
        ],
        entities=[
            "FAISS", "Pinecone", "Weaviate", "pgvector", "Chroma", "HNSW", "IVF-PQ", "PostgreSQL",
        ],
        summary=(
            "Benchmark and operational evaluation comparing FAISS, Pinecone, Weaviate, pgvector, and Chroma. "
            "Covers performance latency, index builds, metadata filtering, memory footprints, and cost trade-offs."
        ),
        suitable_queries=[
            "Which vector database provides lowest latency for 1M vectors?",
            "Compare Pinecone and Weaviate on cost and scalability.",
            "How does pgvector handle vector indexing compared to dedicated databases?",
        ],
    ),
    "agentic_ai_frameworks.pdf": DocumentMetadata(
        doc_name="agentic_ai_frameworks.pdf",
        title="Agentic AI Frameworks & Multi-Agent Orchestration",
        domain="Agent Orchestration & Workflow Architectures",
        topics=[
            "Comparison of LangChain, LangGraph, CrewAI, and AutoGen",
            "Directed Acyclic Graphs (DAGs) vs cyclic state machines",
            "Agent patterns: ReAct, Plan-and-Solve, Multi-Agent Collaboration, Hierarchical Supervision",
            "Tool execution, sandboxing, and function calling",
            "Human-in-the-loop checkpoints and state persistence",
            "Error handling, replanning, and self-reflection loops",
        ],
        entities=[
            "LangChain", "LangGraph", "CrewAI", "AutoGen", "ReAct", "StateGraph",
            "Human-in-the-loop", "Reflection", "Supervisor Agent",
        ],
        summary=(
            "Detailed comparative architectural review of agent orchestration frameworks: LangChain, LangGraph, "
            "CrewAI, and AutoGen. Covers state management, cyclic loops, tool integration, and multi-agent coordination."
        ),
        suitable_queries=[
            "What is the difference between LangGraph and AutoGen?",
            "How does LangGraph handle cyclic agent workflows?",
            "What multi-agent roles are provided in CrewAI?",
        ],
    ),
}


def get_corpus_context_string() -> str:
    """Format corpus metadata as a prompt-ready context string for the Question Router."""
    lines = ["Available Knowledge Base Documents and Coverage:"]
    for doc in CORPUS_METADATA.values():
        lines.append(f"\nDocument: {doc.doc_name}")
        lines.append(f"  Title: {doc.title}")
        lines.append(f"  Domain: {doc.domain}")
        lines.append(f"  Key Topics: {', '.join(doc.topics)}")
        lines.append(f"  Summary: {doc.summary}")
    return "\n".join(lines)
