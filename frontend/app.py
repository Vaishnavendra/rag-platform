import uuid
import time
import requests
import pandas as pd
import streamlit as st

# ---------------------------------------------------------
# Page Configuration & Styling
# ---------------------------------------------------------
st.set_page_config(
    page_title="Production RAG Platform",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern polished design
st.markdown("""
<style>
    /* Global styling */
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(90deg, #3b82f6 0%, #8b5cf6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        color: #64748b;
        font-size: 1rem;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 1rem;
        text-align: center;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    .badge-status-online {
        background-color: #10b981;
        color: white;
        padding: 2px 10px;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    .badge-status-offline {
        background-color: #ef4444;
        color: white;
        padding: 2px 10px;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    .chunk-box {
        background-color: rgba(59, 130, 246, 0.08);
        border-left: 3px solid #3b82f6;
        padding: 10px 14px;
        border-radius: 4px;
        margin-bottom: 8px;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Session State Initialization
# ---------------------------------------------------------
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = []

if "metrics_history" not in st.session_state:
    st.session_state.metrics_history = []

if "latest_metrics" not in st.session_state:
    st.session_state.latest_metrics = None

if "ingested_docs" not in st.session_state:
    st.session_state.ingested_docs = []

# ---------------------------------------------------------
# Sidebar Configuration
# ---------------------------------------------------------
with st.sidebar:
    st.title("⚡ RAG Control Panel")
    
    api_base_url = st.text_input(
        "Backend API Base URL",
        value="http://127.0.0.1:8000",
        help="FastAPI backend endpoint base URL"
    ).rstrip("/")
    
    # Backend Health Check
    try:
        health_resp = requests.get(f"{api_base_url}/health", timeout=2)
        if health_resp.status_code == 200:
            st.markdown('Backend Status: <span class="badge-status-online">CONNECTED</span>', unsafe_allow_html=True)
            health_data = health_resp.json()
            st.caption(f"Service: `{health_data.get('service', 'RAG API')}`")
        else:
            st.markdown('Backend Status: <span class="badge-status-offline">ERROR</span>', unsafe_allow_html=True)
    except Exception:
        st.markdown('Backend Status: <span class="badge-status-offline">DISCONNECTED</span>', unsafe_allow_html=True)
        st.caption("Ensure FastAPI is running on port 8000.")
    
    st.divider()
    
    # Session Controls
    st.subheader("Session Management")
    st.code(st.session_state.session_id, language="text")
    
    col_s1, col_s2 = st.columns(2)
    with col_s1:
        if st.button("🔄 New Session", use_container_width=True):
            st.session_state.session_id = str(uuid.uuid4())
            st.session_state.messages = []
            st.session_state.latest_metrics = None
            st.rerun()
            
    with col_s2:
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
            
    st.divider()
    
    # Search Parameters
    st.subheader("Retrieval Config")
    top_k = st.slider("Top K Chunks (Retrieval)", min_value=1, max_value=10, value=3, step=1)
    
    st.divider()
    st.caption("Production RAG Platform • v0.1.0")

# ---------------------------------------------------------
# Main App Header & Tabs
# ---------------------------------------------------------
st.markdown('<div class="main-header">Distributed Real-Time RAG Platform</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Document Ingestion, Conversational Multi-turn RAG, and Performance Observability</div>', unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs([
    "📥 Document Ingestion",
    "💬 Conversational RAG Chat",
    "📊 System & Latency Metrics"
])

# ---------------------------------------------------------
# TAB 1: Document Ingestion
# ---------------------------------------------------------
with tab1:
    st.subheader("Ingest Knowledge Documents")
    st.markdown("Add structured text documents to the vector store (`Qdrant`) with automated chunking and embeddings.")
    
    with st.form("ingestion_form", clear_on_submit=False):
        col_t1, col_t2 = st.columns([3, 1])
        with col_t1:
            doc_title = st.text_input(
                "Document Title",
                placeholder="e.g., Q3 Financial Overview / System Architecture Documentation",
                help="Unique or descriptive title for the document."
            )
        with col_t2:
            st.write("")
            st.write("")
            sample_btn = st.form_submit_button("📋 Load Sample Text")
        
        # If user clicked sample load
        default_content = ""
        if sample_btn:
            doc_title = "FastAPI & RAG Architecture"
            default_content = (
                "Retrieval-Augmented Generation (RAG) combines the power of dense vector indexing "
                "with Large Language Models (LLMs) to answer user questions using specific context chunks. "
                "In our distributed architecture, documents are chunked using semantic splitters, converted into embeddings, "
                "and indexed in Qdrant. Multi-turn memory allows rewriting conversational queries into standalone prompts "
                "for robust hybrid vector search."
            )
        
        doc_content = st.text_area(
            "Document Content",
            value=default_content,
            height=200,
            placeholder="Paste raw text, knowledge base content, notes, or articles here...",
            help="Text content that will be chunked and indexed."
        )
        
        submitted = st.form_submit_button("🚀 Ingest Document", type="primary", use_container_width=True)
        
    if submitted:
        if not doc_content.strip():
            st.error("⚠️ Document content cannot be empty!")
        else:
            title_to_send = doc_title.strip() if doc_title.strip() else "Untitled Document"
            payload = {
                "title": title_to_send,
                "content": doc_content.strip()
            }
            
            with st.spinner("Chunking text, creating embeddings, and indexing into Qdrant..."):
                try:
                    response = requests.post(
                        f"{api_base_url}/api/v1/ingest/text",
                        json=payload,
                        timeout=30
                    )
                    
                    if response.status_code == 200:
                        res_data = response.json()
                        st.success(f"✅ Document **'{res_data.get('title')}'** successfully ingested and indexed!")
                        
                        m1, m2, m3 = st.columns(3)
                        m1.metric("Document Title", res_data.get("title"))
                        m2.metric("Total Chunks Created", res_data.get("total_chunks", 0))
                        m3.metric("Vectors Indexed", res_data.get("indexed_vectors", 0))
                        
                        # Save to session history
                        st.session_state.ingested_docs.append({
                            "title": res_data.get("title"),
                            "chunks": res_data.get("total_chunks", 0),
                            "vectors": res_data.get("indexed_vectors", 0),
                            "time": time.strftime("%H:%M:%S")
                        })
                        
                        # Expandable chunks viewer
                        chunks = res_data.get("chunks", [])
                        if chunks:
                            with st.expander(f"🔍 Inspect Generated Chunks ({len(chunks)})"):
                                for i, chunk in enumerate(chunks, 1):
                                    chunk_text = chunk.get("content") if isinstance(chunk, dict) else str(chunk)
                                    char_count = chunk.get("character_count") if isinstance(chunk, dict) else len(str(chunk))
                                    st.markdown(f"**Chunk #{i}** ({char_count} chars)")
                                    st.markdown(f'<div class="chunk-box">{chunk_text}</div>', unsafe_allow_html=True)
                    else:
                        st.error(f"❌ Ingestion Failed (Status {response.status_code}): {response.text}")
                except Exception as e:
                    st.error(f"❌ Connection error while contacting backend at {api_base_url}: {str(e)}")

    if st.session_state.ingested_docs:
        st.divider()
        st.markdown("##### 📚 Ingested Documents in Current Session")
        st.dataframe(pd.DataFrame(st.session_state.ingested_docs), use_container_width=True)

# ---------------------------------------------------------
# TAB 2: Conversational RAG Chat
# ---------------------------------------------------------
with tab2:
    st.subheader("Multi-Turn Conversational Assistant")
    st.caption("Ask questions grounded in your ingested documents. Conversation history is automatically contextualized.")
    
    # Display message history
    for msg in st.session_state.messages:
        role = msg["role"]
        content = msg["content"]
        with st.chat_message(role):
            st.markdown(content)
            if role == "assistant":
                # Show sources if stored
                sources = msg.get("sources", [])
                if sources:
                    with st.expander("📚 View Retrieved Sources & Context", expanded=False):
                        for s_idx, src in enumerate(sources, 1):
                            score = src.get("score")
                            score_text = f"(Score: {score:.4f})" if score is not None else ""
                            src_title = src.get("title") or src.get("document_title") or "Unknown Document"
                            src_text = src.get("content") or src.get("chunk_text") or src.get("text") or ""
                            st.markdown(f"**Source #{s_idx}: {src_title}** {score_text}")
                            st.markdown(f'<div class="chunk-box">{src_text}</div>', unsafe_allow_html=True)
                
                # Show standalone query rewrite if available
                standalone = msg.get("standalone_query")
                if standalone and standalone != msg.get("original_query"):
                    st.caption(f"🧠 *Contextualized query rewrite: \"{standalone}\"*")

    # Chat Input
    if user_prompt := st.chat_input("Ask a question about your knowledge base..."):
        # Display user message immediately
        st.chat_message("user").markdown(user_prompt)
        st.session_state.messages.append({"role": "user", "content": user_prompt})
        
        # Prepare Request
        payload = {
            "query": user_prompt,
            "session_id": st.session_state.session_id,
            "top_k": top_k
        }
        
        with st.chat_message("assistant"):
            with st.spinner("Retrieving relevant context and generating response..."):
                try:
                    # Endpoint specified: POST /genrate_answer/answer
                    resp = requests.post(
                        f"{api_base_url}/genrate_answer/answer",
                        json=payload,
                        timeout=60
                    )
                    
                    if resp.status_code == 200:
                        data = resp.json()
                        answer_text = data.get("answer", "")
                        metrics = data.get("metrices", {})
                        sources = data.get("sources", [])
                        standalone_query = data.get("standalone_query", user_prompt)
                        
                        st.markdown(answer_text)
                        
                        # Show sources
                        if sources:
                            with st.expander("📚 View Retrieved Sources & Context", expanded=False):
                                for s_idx, src in enumerate(sources, 1):
                                    score = src.get("score")
                                    score_text = f"(Score: {score:.4f})" if score is not None else ""
                                    src_title = src.get("title") or src.get("document_title") or "Unknown Document"
                                    src_text = src.get("content") or src.get("chunk_text") or src.get("text") or ""
                                    st.markdown(f"**Source #{s_idx}: {src_title}** {score_text}")
                                    st.markdown(f'<div class="chunk-box">{src_text}</div>', unsafe_allow_html=True)
                        
                        if standalone_query and standalone_query != user_prompt:
                            st.caption(f"🧠 *Contextualized query rewrite: \"{standalone_query}\"*")
                            
                        # Save metrics
                        st.session_state.latest_metrics = metrics
                        st.session_state.metrics_history.append({
                            "timestamp": time.strftime("%H:%M:%S"),
                            "query": user_prompt,
                            "retrieval_ms": metrics.get("retrieval_ms", 0.0),
                            "llm_generation_ms": metrics.get("llm_generation_ms", 0.0),
                            "total_ms": metrics.get("total_ms", 0.0)
                        })
                        
                        # Store in message history
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": answer_text,
                            "sources": sources,
                            "standalone_query": standalone_query,
                            "original_query": user_prompt,
                            "metrics": metrics
                        })
                        
                    else:
                        error_msg = f"❌ API Error ({resp.status_code}): {resp.text}"
                        st.error(error_msg)
                        st.session_state.messages.append({"role": "assistant", "content": error_msg})
                except Exception as e:
                    error_msg = f"❌ Error communicating with backend: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})

# ---------------------------------------------------------
# TAB 3: System & Latency Metrics
# ---------------------------------------------------------
with tab3:
    st.subheader("System Observability & Latency Metrics")
    st.markdown("Real-time telemetry and timing breakdowns returned from the generation engine.")
    
    if st.session_state.latest_metrics:
        m = st.session_state.latest_metrics
        retrieval_ms = m.get("retrieval_ms", 0.0)
        llm_ms = m.get("llm_generation_ms", 0.0)
        total_ms = m.get("total_ms", 0.0)
        
        st.markdown("##### ⚡ Latest Query Latency Breakdown")
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        
        col_m1.metric("Total Latency", f"{total_ms:.2f} ms")
        col_m2.metric("Vector Retrieval", f"{retrieval_ms:.2f} ms", delta=f"{(retrieval_ms/total_ms*100 if total_ms else 0):.1f}% of total", delta_color="off")
        col_m3.metric("LLM Generation", f"{llm_ms:.2f} ms", delta=f"{(llm_ms/total_ms*100 if total_ms else 0):.1f}% of total", delta_color="off")
        col_m4.metric("Active Session ID", st.session_state.session_id[:8] + "...")
        
        # Breakdown Visuals
        st.divider()
        col_c1, col_c2 = st.columns([1, 1])
        
        with col_c1:
            st.markdown("##### 📊 Latency Distribution")
            chart_df = pd.DataFrame({
                "Component": ["Vector Retrieval", "LLM Generation"],
                "Latency (ms)": [retrieval_ms, llm_ms]
            }).set_index("Component")
            st.bar_chart(chart_df, color="#3b82f6")
            
        with col_c2:
            st.markdown("##### 📈 Component Share (%)")
            if total_ms > 0:
                retrieval_pct = (retrieval_ms / total_ms) * 100
                llm_pct = (llm_ms / total_ms) * 100
                other_pct = max(0.0, 100.0 - retrieval_pct - llm_pct)
                pct_df = pd.DataFrame({
                    "Component": ["Retrieval", "LLM Generation", "Overhead / Guardrails"],
                    "Share (%)": [retrieval_pct, llm_pct, other_pct]
                }).set_index("Component")
                st.dataframe(pct_df.style.format("{:.2f}%"), use_container_width=True)
            
        if len(st.session_state.metrics_history) > 1:
            st.divider()
            st.markdown("##### 📉 Session Latency Trend")
            hist_df = pd.DataFrame(st.session_state.metrics_history)
            st.line_chart(hist_df.set_index("timestamp")[["retrieval_ms", "llm_generation_ms", "total_ms"]])
            
        st.divider()
        st.markdown("##### 📜 Historical Queries Telemetry")
        st.dataframe(pd.DataFrame(st.session_state.metrics_history), use_container_width=True)
        
    else:
        st.info("ℹ️ No queries have been executed in this session yet. Run a prompt in Tab 2 to record timing metrics!")
        
        # Placeholder / Empty State preview
        st.markdown("##### 📋 Expected Telemetry Breakdown")
        demo_df = pd.DataFrame({
            "Metric": ["retrieval_ms", "llm_generation_ms", "total_ms"],
            "Description": [
                "Time taken by Qdrant hybrid vector search to fetch context chunks",
                "Time taken by LLM (Groq) to generate final answer from context",
                "Total end-to-end request processing time including guardrails and contextualization"
            ]
        })
        st.dataframe(demo_df, use_container_width=True)
