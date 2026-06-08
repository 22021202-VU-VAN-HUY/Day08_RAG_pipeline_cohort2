"""
Group Project — RAG Chatbot với Hybrid Search + Generation + Conversation Memory

Chạy:
    streamlit run group_project/app.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import os
import streamlit as st
from dotenv import load_dotenv

load_dotenv(ROOT / ".env")

from src.task9_retrieval_pipeline import retrieve
from src.task10_generation import (
    reorder_for_llm,
    format_context,
    SYSTEM_PROMPT,
    TEMPERATURE,
    TOP_P,
)

# =============================================================================
# PAGE CONFIG
# =============================================================================
st.set_page_config(page_title="DrugLaw Chatbot", page_icon="🤖", layout="wide")

# =============================================================================
# SIDEBAR
# =============================================================================
with st.sidebar:
    st.header("⚙️ Settings")

    top_k = st.slider("Top K chunks", 1, 20, 5)
    score_threshold = st.slider("Score Threshold", 0.0, 1.0, 0.3, 0.05)
    use_reranking = st.checkbox("Cross-Encoder Reranking", value=True)

    st.divider()

    mimo_key = os.getenv("MIMO_API_KEY", "")
    mimo_url = os.getenv("MIMO_BASE_URL", "")
    if mimo_key and mimo_url:
        st.success("Mimo API: Connected")
    else:
        st.error("Mimo API: Not configured\n\nAdd MIMO_API_KEY + MIMO_BASE_URL to .env")

    st.divider()

    if st.button("🗑️ Xóa lịch sử chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.divider()
    st.header("📊 Index Stats")
    try:
        from src.task4_chunking_indexing import get_chroma_collection
        col = get_chroma_collection()
        st.metric("Chunks Indexed", col.count())
    except Exception:
        st.warning("Chưa index data")

# =============================================================================
# SESSION STATE
# =============================================================================
if "messages" not in st.session_state:
    st.session_state.messages = []

# =============================================================================
# HEADER
# =============================================================================
st.title("🤖 DrugLaw Chatbot")
st.caption("Hỏi đáp pháp luật về ma tuý & tin tức nghệ sĩ | Hybrid RAG + Mimo LLM")

# =============================================================================
# DISPLAY CHAT HISTORY
# =============================================================================
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander(f"📚 Sources ({len(msg['sources'])} chunks)"):
                for i, src in enumerate(msg["sources"], 1):
                    s = src.get("metadata", {}).get("source", "?")
                    t = src.get("metadata", {}).get("type", "?")
                    score = src.get("score", 0)
                    emoji = "📜" if t == "legal" else "📰"
                    st.markdown(f"**{emoji} [{i}] {s}** (score: {score:.4f})")
                    st.caption(src["content"][:200] + "...")

# =============================================================================
# CHAT INPUT
# =============================================================================
query = st.chat_input("Hỏi về pháp luật ma tuý, nghệ sĩ liên quan...")

if query:
    # Show user message
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    # Generate answer
    with st.chat_message("assistant"):
        with st.spinner("Đang tìm kiếm và tạo câu trả lời..."):
            # Step 1: Retrieve
            chunks = retrieve(
                query,
                top_k=top_k,
                score_threshold=score_threshold,
                use_reranking=use_reranking,
            )

            if not chunks:
                answer = "Tôi không tìm thấy thông tin liên quan trong cơ sở dữ liệu. Vui lòng thử câu hỏi khác."
                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
            else:
                # Step 2: Reorder
                reordered = reorder_for_llm(chunks)

                # Step 3: Format context
                context = format_context(reordered)

                # Step 4: Build messages with conversation history
                messages_for_llm = [{"role": "system", "content": SYSTEM_PROMPT}]

                # Add conversation history (last 6 messages for context)
                for msg in st.session_state.messages[-6:]:
                    messages_for_llm.append({
                        "role": msg["role"],
                        "content": msg["content"],
                    })

                # Add current query with context
                user_msg = f"Context:\n{context}\n\n---\n\nQuestion: {query}"
                messages_for_llm.append({"role": "user", "content": user_msg})

                # Step 5: Call LLM
                try:
                    from openai import OpenAI

                    mimo_key = os.getenv("MIMO_API_KEY", "")
                    mimo_base = os.getenv("MIMO_BASE_URL", "")
                    mimo_model = os.getenv("MIMO_MODEL", "mimo-v2.5-pro")

                    if mimo_key and mimo_base:
                        client = OpenAI(api_key=mimo_key, base_url=mimo_base)
                        model = mimo_model
                    else:
                        raise ValueError("No API key configured")

                    response = client.chat.completions.create(
                        model=model,
                        messages=messages_for_llm,
                        temperature=TEMPERATURE,
                        top_p=TOP_P,
                        max_tokens=1024,
                    )
                    answer = response.choices[0].message.content or ""
                except Exception as e:
                    answer = f"Lỗi gọi LLM: {e}\n\nDưới đây là kết quả tìm kiếm thô:\n\n"
                    for i, c in enumerate(chunks[:3], 1):
                        answer += f"**[{i}]** {c['content'][:300]}...\n\n"

                # Display answer
                st.markdown(answer)

                # Display sources
                with st.expander(f"📚 Sources ({len(chunks)} chunks)"):
                    for i, src in enumerate(chunks, 1):
                        s = src.get("metadata", {}).get("source", "?")
                        t = src.get("metadata", {}).get("type", "?")
                        score = src.get("score", 0)
                        emoji = "📜" if t == "legal" else "📰"
                        st.markdown(f"**{emoji} [{i}] {s}** (score: {score:.4f})")
                        st.caption(src["content"][:200] + "...")

                # Save to history
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": chunks,
                })
