import streamlit as st
from cell_culture_functions import (
    load_knowledge_base,
    create_chunks_with_sources,
    create_vector_store,
    rag_answer_with_sources
)

# ============= PAGE CONFIG =============
st.set_page_config(
    page_title="CODE4CARE - Cell Culture Assistant",
    page_icon="🧬",
    layout="wide"
)

# ============= SIDEBAR =============
with st.sidebar:
    st.title("🧬 CODE4CARE")
    st.markdown("### Cell Culture Assistant")
    st.markdown("---")

    st.markdown("""
    **Skin Cultivation Lab Support**

    This AI assistant helps operators with:
    - Troubleshooting cell culture issues
    - Best practices for skin cultivation
    - Contamination prevention
    - Temperature and incubator management
    - Emergency protocols
    """)

    st.markdown("---")

    # Initialize knowledge base button
    if st.button("🔄 Load/Reload Knowledge Base", use_container_width=True):
        with st.spinner("Loading knowledge base..."):
            docs, error = load_knowledge_base()
            if error:
                st.error(error)
            elif docs:
                chunks = create_chunks_with_sources(docs)
                vectorstore = create_vector_store(chunks)
                st.session_state.vector_store = vectorstore
                st.session_state.messages = []
                st.success(f"✅ Loaded {len(docs)} document(s), created {len(chunks)} chunks")
            else:
                st.error("No documents found in knowledge_base folder")

    st.markdown("---")
    st.markdown("**⚡ Quick Questions:**")

    quick_questions = [
        "What should I do if media turns yellow?",
        "How do I prevent contamination?",
        "What is the optimal temperature?",
        "How do I split cultures safely?",
        "What if incubator fails?"
    ]

    for q in quick_questions:
        if st.button(q, use_container_width=True, key=f"quick_{q}"):
            if "vector_store" in st.session_state:
                st.session_state.pending_question = q

# ============= MAIN AREA =============
st.title("🧬 Cell Culture Lab Assistant")
st.markdown("*Your AI-powered troubleshooting playbook for skin cell cultivation*")

# Check if knowledge base is loaded
if "vector_store" not in st.session_state:
    st.info("👈 Click 'Load/Reload Knowledge Base' in the sidebar to get started")

    # Auto-load on first run
    with st.spinner("Auto-loading knowledge base..."):
        docs, error = load_knowledge_base()
        if error:
            st.error(error)
        elif docs:
            chunks = create_chunks_with_sources(docs)
            vectorstore = create_vector_store(chunks)
            st.session_state.vector_store = vectorstore
            st.session_state.messages = []
            st.success(f"✅ Knowledge base loaded: {len(docs)} document(s), {len(chunks)} searchable chunks")
            st.rerun()

else:
    st.divider()

    # Initialize messages
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Handle pending question from sidebar
    if "pending_question" in st.session_state:
        prompt = st.session_state.pending_question
        del st.session_state.pending_question

        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("🔍 Searching knowledge base..."):
                response = rag_answer_with_sources(prompt, st.session_state.vector_store)
            st.markdown(response)

        st.session_state.messages.append({"role": "assistant", "content": response})
        st.rerun()

    # Chat input
    prompt = st.chat_input("Ask about cell culture, contamination, temperature, troubleshooting...")

    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("🔍 Searching knowledge base..."):
                response = rag_answer_with_sources(prompt, st.session_state.vector_store)
            st.markdown(response)

        st.session_state.messages.append({"role": "assistant", "content": response})
