"""
capstone_streamlit.py — HR Policy Bot Streamlit UI
Run: streamlit run capstone_streamlit.py
Opens browser at: http://localhost:8501

Agentic AI Capstone Project | HR Policy Bot
Author: [Your Name] | Roll: [Your Roll No] | Batch: Agentic AI 2026
"""

import streamlit as st
import uuid

# ── Page config ─────────────────────────────────────────────
st.set_page_config(
    page_title="HR Policy Bot",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS for a professional look ──────────────────────
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #1e3a5f 0%, #2d6a9f 100%);
        padding: 1.5rem 2rem;
        border-radius: 10px;
        margin-bottom: 1rem;
    }
    .main-header h1 { color: white; margin: 0; font-size: 2rem; }
    .main-header p  { color: #b0cfe8; margin: 0.3rem 0 0 0; }
    .sidebar-section {
        background: #f0f4f8;
        border-radius: 8px;
        padding: 1rem;
        margin-bottom: 1rem;
    }
    .chat-source {
        font-size: 0.75rem;
        color: #6b7a8d;
        border-top: 1px solid #e0e7ef;
        margin-top: 0.5rem;
        padding-top: 0.3rem;
    }
    .metric-badge {
        background: #e8f4fd;
        border: 1px solid #90caf9;
        border-radius: 5px;
        padding: 0.2rem 0.6rem;
        font-size: 0.8rem;
        color: #1565c0;
        display: inline-block;
        margin: 0.2rem 0.1rem;
    }
</style>
""", unsafe_allow_html=True)


# ── Load agent (cached so it only initialises once) ─────────
@st.cache_resource
def load_agent():
    """Load the agent once and cache it — prevents re-init on every rerun."""
    from agent import ask, DOCUMENTS, collection
    return ask, DOCUMENTS, collection


ask_fn, DOCUMENTS, collection = load_agent()


# ── Session state initialisation ─────────────────────────────
if "messages"   not in st.session_state:
    st.session_state.messages   = []
if "thread_id"  not in st.session_state:
    st.session_state.thread_id  = str(uuid.uuid4())
if "user_name"  not in st.session_state:
    st.session_state.user_name  = ""
if "last_meta"  not in st.session_state:
    st.session_state.last_meta  = {}


# ── Sidebar ──────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏢 HR Policy Bot")
    st.markdown("**Agentic AI Capstone Project**")
    st.divider()

    st.markdown("### 📋 About")
    st.markdown("""
This intelligent HR assistant helps company employees instantly access HR policies 24/7.
Ask about **leave, payroll, benefits, PF, resignation**, and more.
    """)

    st.divider()
    st.markdown("### 📚 Topics Covered")
    topics = [d["topic"] for d in DOCUMENTS]
    for topic in topics:
        st.markdown(f"- {topic}")

    st.divider()
    st.markdown("### 💡 Sample Questions")
    sample_qs = [
        "How many annual leave days do I get?",
        "What is the notice period for resignation?",
        "How is gratuity calculated?",
        "What does health insurance cover?",
        "What is today's date?",
    ]
    for q in sample_qs:
        if st.button(q, key=f"sample_{q[:20]}", use_container_width=True):
            st.session_state.pending_question = q

    st.divider()
    if st.button("🔄 New Conversation", use_container_width=True, type="primary"):
        st.session_state.messages  = []
        st.session_state.thread_id = str(uuid.uuid4())
        st.session_state.user_name = ""
        st.session_state.last_meta = {}
        st.rerun()

    st.divider()
    st.markdown(f"**Session ID:** `{st.session_state.thread_id[:8]}...`")
    if st.session_state.user_name:
        st.markdown(f"**Employee:** {st.session_state.user_name}")

    st.markdown("---")
    st.markdown("📞 **HR Helpdesk:** 1800-HR-HELP")
    st.markdown("📧 hr-helpdesk@company.com")


# ── Main header ──────────────────────────────────────────────
st.markdown("""
<div class="main-header">
  <h1>🏢 HR Policy Bot</h1>
  <p>Your 24/7 AI assistant for HR policies, leave, payroll & benefits — powered by LangGraph + RAG</p>
</div>
""", unsafe_allow_html=True)

# ── Render chat history ───────────────────────────────────────
for i, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"], avatar="👤" if msg["role"] == "user" else "🤖"):
        st.markdown(msg["content"])
        # Show metadata for assistant messages
        if msg["role"] == "assistant" and msg.get("sources"):
            sources_str = " | ".join(msg["sources"])
            faith_str   = f"  Faithfulness: {msg['faithfulness']:.2f}" if msg.get("faithfulness") else ""
            st.markdown(
                f'<div class="chat-source">📂 Sources: {sources_str}{faith_str}</div>',
                unsafe_allow_html=True
            )

# ── Handle sample question clicks ────────────────────────────
if "pending_question" in st.session_state:
    pending = st.session_state.pop("pending_question")
    st.session_state.messages.append({"role": "user", "content": pending})
    with st.chat_message("user", avatar="👤"):
        st.markdown(pending)
    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner("Searching HR policies..."):
            result = ask_fn(pending, thread_id=st.session_state.thread_id)
        answer  = result.get("answer", "")
        sources = result.get("sources", [])
        faith   = result.get("faithfulness", 1.0)
        if result.get("user_name"):
            st.session_state.user_name = result["user_name"]
        st.markdown(answer)
        if sources:
            st.markdown(
                f'<div class="chat-source">📂 Sources: {" | ".join(sources)}  '
                f'Faithfulness: {faith:.2f}</div>',
                unsafe_allow_html=True
            )
        st.session_state.messages.append({
            "role": "assistant", "content": answer,
            "sources": sources, "faithfulness": faith
        })

# ── Chat input ────────────────────────────────────────────────
if prompt := st.chat_input("Ask about leave, payroll, benefits, or any HR policy..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="👤"):
        st.markdown(prompt)

    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner("🔍 Searching HR policy database..."):
            result = ask_fn(prompt, thread_id=st.session_state.thread_id)

        answer  = result.get("answer", "")
        sources = result.get("sources", [])
        faith   = result.get("faithfulness", 1.0)
        route   = result.get("route", "retrieve")

        if result.get("user_name"):
            st.session_state.user_name = result["user_name"]

        st.markdown(answer)

        if sources:
            st.markdown(
                f'<div class="chat-source">📂 Sources: {" | ".join(sources)}'
                f'  •  Route: {route}  •  Faithfulness: {faith:.2f}</div>',
                unsafe_allow_html=True
            )

        st.session_state.messages.append({
            "role": "assistant", "content": answer,
            "sources": sources, "faithfulness": faith
        })

# ── Footer ────────────────────────────────────────────────────
if not st.session_state.messages:
    st.markdown("""
    <div style="text-align:center; padding:2rem; color:#6b7a8d;">
        <p style="font-size:1.1rem;">👋 Hello! I'm your HR Policy Assistant.</p>
        <p>Ask me about <b>leave policies</b>, <b>payroll</b>, <b>benefits</b>,
        <b>PF/gratuity</b>, <b>resignation process</b>, and more.</p>
        <p style="font-size:0.85rem;">Try: <i>"How many annual leave days do I get?"</i>
        or <i>"What is the notice period for resignation?"</i></p>
    </div>
    """, unsafe_allow_html=True)
