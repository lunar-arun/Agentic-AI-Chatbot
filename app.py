"""
app.py — Streamlit UI for the multi-agent customer-service chatbot
with AGENTIC RAG.

Run with:
    streamlit run app.py

Add your own .txt or .md files to the knowledge_base/ folder to
extend the chatbot's knowledge.  Changes are detected automatically on
the next restart or when you press "Re-scan knowledge base" in the sidebar
(SHA-256 hash comparison).
"""

import streamlit as st
from agents import (
    run_agent1_loop,
    stream_agent1_answer,
    stream_agent2,
    GENERATION_MODEL,
    RELEVANCE_THRESHOLD,
    MAX_RETRIEVAL_ITERATIONS,
)
from rag import initialize_knowledge_base, EMBEDDING_MODEL
from validation import filter_input, escape_html, inline_code_safe

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="GreenTech Customer Support",
    page_icon="🌿",
    layout="centered",
)

# ---------------------------------------------------------------------------
# Custom CSS  (the dark base theme itself is set in .streamlit/config.toml)
# ---------------------------------------------------------------------------

st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #0f2027 0%, #203a43 50%, #2c5364 100%);
    }
    .stChatMessage {
        border-radius: 12px !important;
        margin-bottom: 0.5rem !important;
        background: rgba(255, 255, 255, 0.05) !important;
    }
    .agent-badge {
        display: inline-block;
        padding: 2px 10px;
        border-radius: 12px;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.03em;
        margin-bottom: 4px;
    }
    .agent-badge.agent1 {
        background: linear-gradient(90deg, #00c9ff, #92fe9d);
        color: #0f2027;
    }
    .agent-badge.agent2 {
        background: linear-gradient(90deg, #fc5c7d, #6a82fb);
        color: #fff;
    }
    .chunk-preview {
        color: #b0bec5;
        font-size: 0.8rem;
        overflow-wrap: anywhere;
    }
    section[data-testid="stSidebar"] {
        background: rgba(15, 32, 39, 0.95);
    }
    @media (max-width: 640px) {
        .block-container { padding-left: 1rem; padding-right: 1rem; }
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []
    # Each message:
    # {"role": "user"|"assistant", "content": str,
    #  "agent": str|None, "trace": list|None, "handoff_reason": str|None}

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("## 🌿 GreenTech Support")
    st.caption(f"Model: `{GENERATION_MODEL}`")
    st.caption(f"Embeddings: `{EMBEDDING_MODEL}`")
    st.caption(f"Relevance threshold: `{RELEVANCE_THRESHOLD}`")
    st.caption(f"Max retrieval iterations: `{MAX_RETRIEVAL_ITERATIONS}`")
    st.markdown("---")
    st.markdown(
        "**How it works — Agentic RAG**\n\n"
        "1. Your message is checked for spam/junk.\n"
        "2. **Agent 1** decides: retrieve, answer, or handoff.\n"
        "3. If retrieving: queries the KB, scores chunks, judges sufficiency.\n"
        f"4. Refines and retries if results are insufficient (up to "
        f"{MAX_RETRIEVAL_ITERATIONS}×).\n"
        "5. Streams the final answer, or escalates to **Agent 2**.\n"
    )
    st.markdown("---")
    st.markdown(
        "📁 Add `.txt` or `.md` files to `knowledge_base/`, then press "
        "the button below (or restart) to expand the chatbot's knowledge."
    )
    if st.button("🔄 Re-scan knowledge base"):
        st.session_state.kb_ok = False  # forces re-initialisation below
    if st.button("🗑️ Clear conversation"):
        st.session_state.messages = []
        st.rerun()

# ---------------------------------------------------------------------------
# Ollama connectivity check (runs first: the knowledge base needs Ollama too)
# ---------------------------------------------------------------------------

def _model_available(wanted: str, installed: set[str]) -> bool:
    if wanted in installed:
        return True
    return ":" not in wanted and f"{wanted}:latest" in installed


def _check_ollama() -> tuple[bool, str]:
    """Verify Ollama is reachable and the required models are pulled."""
    try:
        import ollama as _ol
        listing = _ol.list()
    except Exception:
        return False, (
            "🚫 **Cannot connect to Ollama.**\n\n"
            "Make sure Ollama is installed and running:\n"
            "```\nollama serve\n```\n"
            "Then pull the required models:\n"
            f"```\nollama pull {GENERATION_MODEL}\n"
            f"ollama pull {EMBEDDING_MODEL}\n```"
        )

    installed = {
        getattr(m, "model", None) or getattr(m, "name", None) or ""
        for m in getattr(listing, "models", [])
    }
    missing = [
        m for m in (GENERATION_MODEL, EMBEDDING_MODEL)
        if not _model_available(m, installed)
    ]
    if missing:
        pulls = "\n".join(f"ollama pull {m}" for m in missing)
        return False, (
            f"🚫 **Missing Ollama model(s):** {', '.join(missing)}\n\n"
            f"```\n{pulls}\n```"
        )
    return True, ""


ollama_ok, ollama_err = _check_ollama()
if not ollama_ok:
    st.error(ollama_err)
    st.stop()

# ---------------------------------------------------------------------------
# Knowledge-base initialization
# (only a SUCCESS is remembered, so a failure is retried on the next rerun)
# ---------------------------------------------------------------------------

if not st.session_state.get("kb_ok"):
    try:
        with st.spinner("🔍 Building knowledge base …"):
            kb_ok, kb_msg = initialize_knowledge_base()
    except Exception as e:
        kb_ok, kb_msg = False, f"Failed to initialize knowledge base: {e}"
    st.session_state.kb_ok = kb_ok
    st.session_state.kb_msg = kb_msg

kb_ok = st.session_state.get("kb_ok", False)
kb_msg = st.session_state.get("kb_msg", "")

if not kb_ok:
    st.warning(kb_msg)
else:
    st.sidebar.success(kb_msg)

# ---------------------------------------------------------------------------
# Reasoning trace renderer
# ---------------------------------------------------------------------------

def _render_trace(trace: list[dict], handoff_reason: str | None = None):
    """Render the agentic RAG reasoning trace in an expander."""
    if not trace and not handoff_reason:
        return

    with st.expander("🧠 Agent 1 reasoning trace"):
        for entry in trace:
            action = entry.get("action", "")
            iteration = entry.get("iteration", "?")

            if "RETRIEVE" in action:
                query = inline_code_safe(entry.get("query", "N/A"))
                chunks = entry.get("chunks", [])
                threshold = entry.get("threshold", RELEVANCE_THRESHOLD)
                judgment = entry.get("judgment", "N/A")

                st.markdown(f"**Iteration {iteration}** — Query: `{query}`")
                for c in chunks:
                    cleared = "✅" if c["score"] >= threshold else "❌"
                    st.markdown(
                        f"&nbsp;&nbsp;{cleared} **{c['score']:.2f}** — "
                        f"`{inline_code_safe(c['source'])}`"
                    )
                    # Preview of the chunk text — escaped, since it is
                    # rendered as raw HTML.
                    preview = c["text"][:200].replace("\n", " ")
                    if len(c["text"]) > 200:
                        preview += "…"
                    st.markdown(
                        f'<div class="chunk-preview">{escape_html(preview)}</div>',
                        unsafe_allow_html=True,
                    )
                st.markdown(f"&nbsp;&nbsp;*{judgment}*")
                st.markdown("---")

            elif "HANDOFF" in action:
                detail = entry.get("detail", "")
                st.markdown(
                    f"**Iteration {iteration}** — 🔀 **HANDOFF**\n\n"
                    f"&nbsp;&nbsp;{detail}"
                )
                st.markdown("---")

            elif "ERROR" in action:
                detail = entry.get("detail", "")
                st.markdown(
                    f"**Iteration {iteration}** — ⚠️ **ERROR**\n\n"
                    f"&nbsp;&nbsp;{detail}"
                )
                st.markdown("---")

            else:  # ANSWER and anything unexpected
                detail = entry.get("detail", "")
                st.markdown(
                    f"**Iteration {iteration}** — ✅ **{action}**\n\n"
                    f"&nbsp;&nbsp;{detail}"
                )
                st.markdown("---")

        if handoff_reason:
            st.markdown(f"**Handoff reason:** {handoff_reason}")


# ---------------------------------------------------------------------------
# Chat history display
# ---------------------------------------------------------------------------

st.markdown("# 🌿 GreenTech Customer Support")
st.caption("Ask about shipping, returns, tracking, billing, or anything else!")

for msg in st.session_state.messages:
    role = msg["role"]
    with st.chat_message(role):
        if role == "assistant" and msg.get("agent"):
            badge_cls = "agent1" if msg["agent"] == "Agent 1" else "agent2"
            st.markdown(
                f'<span class="agent-badge {badge_cls}">'
                f'{escape_html(msg["agent"])}</span>',
                unsafe_allow_html=True,
            )
        st.markdown(msg["content"])

        # Reasoning trace (for Agent 1 responses)
        if role == "assistant" and msg.get("trace"):
            _render_trace(msg["trace"], msg.get("handoff_reason"))


# ---------------------------------------------------------------------------
# Chat input + agent execution
# ---------------------------------------------------------------------------

if user_input := st.chat_input("Type your message …"):

    # ---- Input filtering ----
    rejection = filter_input(user_input)
    if rejection:
        st.warning(rejection)
        st.stop()

    # ---- Show user message ----
    st.session_state.messages.append({
        "role": "user", "content": user_input,
        "agent": None, "trace": None, "handoff_reason": None,
    })
    with st.chat_message("user"):
        st.markdown(user_input)

    # ---- Agent 1: Agentic RAG loop (non-streaming decision phase) ----
    with st.chat_message("assistant"):
        st.markdown(
            '<span class="agent-badge agent1">Agent 1</span>',
            unsafe_allow_html=True,
        )

        # Run the agentic RAG loop — this may take multiple LLM calls
        with st.spinner("🤔 Agent 1 is reasoning …"):
            try:
                result = run_agent1_loop(
                    user_input,
                    st.session_state.messages[:-1],  # history before this turn
                )
            except Exception as e:
                result = {
                    "action": "answer",
                    "context": [],
                    "trace": [{"iteration": 1, "action": "ERROR",
                               "detail": str(e)}],
                    "direct_answer": f"❌ Error in Agent 1: {e}",
                    "reason": "",
                }

        # Show reasoning trace
        _render_trace(
            result["trace"],
            result.get("reason") if result["action"] == "handoff" else None,
        )

        # ---- Stream the final answer (or show direct answer) ----
        if result["action"] == "answer":
            placeholder = st.empty()
            agent1_full = ""

            if result.get("direct_answer"):
                # Direct answer (no retrieval — e.g., greeting)
                agent1_full = result["direct_answer"]
                placeholder.markdown(agent1_full)
            else:
                # Stream answer using retrieved context
                try:
                    for token in stream_agent1_answer(
                        user_input,
                        st.session_state.messages[:-1],
                        result["context"],
                    ):
                        agent1_full += token
                        placeholder.markdown(agent1_full + "▌")
                    placeholder.markdown(agent1_full)
                except Exception as e:
                    agent1_full = f"❌ Error streaming Agent 1 answer: {e}"
                    placeholder.error(agent1_full)

            st.session_state.messages.append({
                "role": "assistant",
                "content": agent1_full,
                "agent": "Agent 1",
                "trace": result["trace"],
                "handoff_reason": None,
            })

        elif result["action"] == "handoff":
            st.info(f"🔀 Escalating to Agent 2 — {result['reason']}")

            st.session_state.messages.append({
                "role": "assistant",
                "content": f"_Escalating to Agent 2: {result['reason']}_",
                "agent": "Agent 1",
                "trace": result["trace"],
                "handoff_reason": result["reason"],
            })

    # ---- Agent 2 (if handoff) ----
    if result["action"] == "handoff":
        with st.chat_message("assistant"):
            st.markdown(
                '<span class="agent-badge agent2">Agent 2</span>',
                unsafe_allow_html=True,
            )
            placeholder = st.empty()
            agent2_full = ""

            try:
                for token in stream_agent2(
                    result["reason"],
                    st.session_state.messages,
                ):
                    agent2_full += token
                    placeholder.markdown(agent2_full + "▌")
                placeholder.markdown(agent2_full)
            except Exception as e:
                agent2_full = f"❌ Error from Agent 2: {e}"
                placeholder.error(agent2_full)

            # No trace here: it is already stored on Agent 1's message.
            st.session_state.messages.append({
                "role": "assistant",
                "content": agent2_full,
                "agent": "Agent 2",
                "trace": None,
                "handoff_reason": None,
            })
