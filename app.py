"""Streamlit demonstration interface for the Intelligent Knowledge Assistant."""
import time
import streamlit as st

from src.config import llm_config, router_config
from src.rag.metadata import CORPUS_METADATA
from src.rag.service import KnowledgeAssistantService

# Page configuration
st.set_page_config(
    page_title="Intelligent Knowledge Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for styling
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .metric-badge-simple {
        background-color: #E0F2FE;
        color: #0369A1;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 600;
        display: inline-block;
    }
    .metric-badge-agentic {
        background-color: #F3E8FF;
        color: #7E22CE;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 600;
        display: inline-block;
    }
    .citation-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 12px;
        margin-bottom: 10px;
    }
    .source-tag {
        font-weight: 600;
        color: #2563EB;
    }
    .muted-container {
        color: #475569;
        background-color: #F8FAFC;
        padding: 12px;
        border-radius: 8px;
        border: 1px solid #E2E8F0;
        font-size: 0.93rem;
    }
    .step-box-muted {
        border-left: 3px solid #94A3B8;
        background-color: #F1F5F9;
        color: #334155;
        padding: 8px 12px;
        margin-bottom: 8px;
        border-radius: 0 6px 6px 0;
        font-size: 0.9rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def get_cached_service() -> KnowledgeAssistantService:
    """Cache the KnowledgeAssistantService in memory to avoid reloading indexes on every rerun."""
    return KnowledgeAssistantService()


# Initialize service
service = get_cached_service()

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")
    
    st.markdown("### 📚 Knowledge Base Corpus")
    for doc_name, meta in CORPUS_METADATA.items():
        with st.expander(f"📄 {doc_name}", expanded=False):
            st.caption(f"**Domain:** {meta.domain}")
            st.caption(f"**Summary:** {meta.summary}")
            st.markdown("**Topics:**")
            for t in meta.topics[:3]:
                st.markdown(f"- {t}")

    st.markdown("---")
    st.markdown("### 🔀 Workflow Policy")
    force_choice = st.radio(
        "Routing Policy",
        options=["auto", "simple_rag", "agentic"],
        format_func=lambda x: {
            "auto": "⚡ Auto (Dynamic Scoring Router)",
            "simple_rag": "📖 Force Simple RAG",
            "agentic": "🧠 Force Agentic AI",
        }[x],
        index=0,
    )

    st.markdown("---")
    st.markdown("### 🤖 Models & Providers")
    st.caption(f"**Primary Model:** `{llm_config.primary_model}`")
    st.caption(f"**Complex Model:** `{llm_config.complex_model}`")
    st.caption(f"**Router Threshold:** `Score >= {router_config.agentic_threshold}`")


# Main interface
st.markdown('<div class="main-header">🤖 Intelligent Knowledge Assistant</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Production RAG & Agentic Multi-Step Reasoning with Question Complexity Scoring & Claim Verification</div>',
    unsafe_allow_html=True,
)

# Demo quick-select questions
sample_questions = [
    "Select a pre-built sample question...",
    "[Simple RAG] What is Reciprocal Rank Fusion?",
    "[Simple RAG] What are the common chunking strategies for technical PDFs?",
    "[Agentic AI] Compare FAISS and Pinecone on latency and cost trade-offs.",
    "[Agentic AI] Compare LangGraph and AutoGen for state management and cyclic workflows.",
    "[Agentic AI] What architectural considerations should I address when deploying a RAG system at scale?",
    "[Unanswerable] What are the clinical treatment guidelines for Type 2 diabetes?",
]

selected_sample = st.selectbox("💡 Try a sample question from the assessment evaluation set:", sample_questions)

default_query = ""
if selected_sample and not selected_sample.startswith("Select"):
    default_query = selected_sample.split("] ", 1)[-1]

# Query input
query_input = st.text_area(
    "Enter your technical question:",
    value=default_query,
    height=90,
    placeholder="Ask anything over RAG patterns, vector databases, or agentic frameworks...",
)

col_btn, col_info = st.columns([1, 4])
with col_btn:
    ask_button = st.button("🚀 Ask Assistant", type="primary", use_container_width=True)

if ask_button and query_input.strip():
    with st.spinner("Analyzing question complexity and retrieving evidence..."):
        start_time = time.time()
        result = service.answer_query(
            query=query_input.strip(),
            force_workflow=force_choice,
        )
        elapsed = time.time() - start_time

    st.markdown("---")

    # Metrics Summary Bar
    col1, col2, col3, col4 = st.columns([2, 2, 2, 2])
    with col1:
        if result.workflow == "agentic":
            st.markdown(
                '<span class="metric-badge-agentic">🧠 Workflow: Agentic AI</span>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<span class="metric-badge-simple">📖 Workflow: Simple RAG</span>',
                unsafe_allow_html=True,
            )

    with col2:
        st.metric("Router Complexity Score", f"{result.routing.total_score} pts", delta=f"Threshold: {result.routing.threshold}")

    with col3:
        status_label = "✅ Supported" if result.verification.status == "supported" else "⚠️ " + result.verification.status.title()
        st.metric("Fact Verification", status_label, delta=f"{int(result.confidence * 100)}% confidence")

    with col4:
        st.metric("Latency", f"{round(result.latency_seconds, 2)}s", delta=f"Model: {result.model_used.split('/')[-1]}")

    st.markdown("---")

    # 1. MAIN ANSWER (Displayed prominently right below the question)
    st.markdown("### 💬 Answer")
    if result.abstained:
        st.warning(result.answer)
    else:
        st.markdown(result.answer)

    st.markdown("---")

    # 2. CITATIONS & SOURCES (Displayed immediately after answer)
    st.markdown(f"### 📑 Cited Sources ({len(result.sources)})")
    if result.sources:
        for i, src in enumerate(result.sources, 1):
            expander_title = f"[{i}] {src.doc_name} (p.{src.page_number} — {src.section})"
            with st.expander(expander_title, expanded=False):
                st.markdown(f"**Relevance Score:** `{src.score}` | **Chunk ID:** `{src.chunk_id}` | **Page:** `{src.page_number}`")
                st.markdown(f"**Section:** `{src.section}`")
                st.markdown("**Excerpt:**")
                st.info(f'"{src.snippet}"')
    else:
        st.caption("No sources cited (System abstained or no chunks passed threshold).")

    st.markdown("---")

    # 3. THINKING, ROUTER & AUDIT DETAILS (Subtle, greyed out, collapsed by default)
    st.markdown("#### 🔬 Execution & Verification Audit (Expand to view)")

    # 3a. Agent Thinking & Decomposition Trace (Collapsed, Muted)
    if result.agent_trace:
        with st.expander("🧠 Agent Thinking, Strategy & Execution Trace", expanded=False):
            st.markdown(
                f"""
                <div class="muted-container">
                    <strong>🎯 Strategic Plan:</strong> {result.agent_trace.plan.strategy}<br/>
                    <strong>Domains Involved:</strong> {', '.join(result.agent_trace.plan.involved_domains)}<br/>
                    <strong>Total Rewrites:</strong> {result.agent_trace.total_rewrites} | 
                    <strong>Unique Chunks Aggregated:</strong> {result.agent_trace.total_unique_chunks}
                </div>
                <br/>
                """,
                unsafe_allow_html=True,
            )
            st.markdown("**Decomposed Sub-Questions & ReAct Retrieval Loops:**")
            for sub in result.agent_trace.sub_executions:
                st.markdown(
                    f"""
                    <div class="step-box-muted">
                        <strong>Sub-Question {sub.step_num}:</strong> {sub.sub_question}<br/>
                        <em>Active Query:</em> <code>{sub.active_query}</code> (Attempts: {sub.attempts})<br/>
                        <em>Sufficiency:</em> {"✅ Sufficient" if sub.is_sufficient else "⚠️ Insufficient"} — {sub.assessment_notes}<br/>
                        {"<em>Rewrites:</em> " + " ➔ ".join(sub.rewrites) if sub.rewrites else ""}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    # 3b. Router Rubric Scoring Breakdown (Collapsed, Muted)
    with st.expander("📊 Router Complexity Scoring Breakdown", expanded=False):
        st.markdown(
            f"""
            <div class="muted-container">
                <strong>Router Rationale:</strong> {result.routing.reasoning}<br/>
                <strong>Predicted Relevant Docs:</strong> <code>{', '.join(result.routing.relevant_docs) if result.routing.relevant_docs else 'None'}</code>
            </div>
            <br/>
            """,
            unsafe_allow_html=True,
        )
        rubric_data = [
            {
                "Feature": f.feature,
                "Detected": "✅ Yes" if f.present else "❌ No",
                "Points": f"+{f.score}" if f.present else "0",
                "Rationale": f.rationale,
            }
            for f in result.routing.features
        ]
        st.table(rubric_data)

    # 3c. Claim Verification & Audit Report (Collapsed, Muted)
    with st.expander("🛡️ Claim Verification & Hallucination Audit Report", expanded=False):
        st.markdown(
            f"""
            <div class="muted-container">
                <strong>Audit Status:</strong> <code>{result.verification.status}</code> | 
                <strong>Grounded:</strong> <code>{result.verification.is_grounded}</code><br/>
                <strong>Audit Notes:</strong> {result.verification.verdict_rationale}
            </div>
            <br/>
            """,
            unsafe_allow_html=True,
        )
        if result.verification.supported_claims:
            st.markdown("**Supported Claims:**")
            for c in result.verification.supported_claims:
                st.markdown(f"- ✅ {c}")
        if result.verification.unsupported_claims:
            st.markdown("**Unsupported / Unverified Statements:**")
            for c in result.verification.unsupported_claims:
                st.markdown(f"- ⚠️ {c}")
