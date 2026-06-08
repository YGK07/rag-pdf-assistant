import streamlit as st
import os

from rag import (
    process_pdfs,
    create_vector_store,
    load_llm,
    ask_rag
)

# -----------------------------
# Page Config
# -----------------------------

st.set_page_config(
    page_title="RAG PDF Assistant",
    page_icon="📚",
    layout="wide"
)

st.title("📚 Multi-PDF RAG Assistant")

# -----------------------------
# Sidebar
# -----------------------------

st.sidebar.header("Upload PDFs")

uploaded_files = st.sidebar.file_uploader(
    "Choose PDF files",
    type=["pdf"],
    accept_multiple_files=True
)

# -----------------------------
# Session State
# -----------------------------

if "documents" not in st.session_state:
    st.session_state.documents = None

if "index" not in st.session_state:
    st.session_state.index = None

if "llm" not in st.session_state:
    st.session_state.llm = load_llm()

if "messages" not in st.session_state:
    st.session_state.messages = []

# -----------------------------
# Process PDFs
# -----------------------------

if st.sidebar.button("Process PDFs"):

    if not uploaded_files:
        st.sidebar.warning(
            "Please upload at least one PDF."
        )

    else:

        upload_folder = "data/uploads"

        os.makedirs(
            upload_folder,
            exist_ok=True
        )

        # Remove old PDFs
        for file in os.listdir(upload_folder):

            file_path = os.path.join(
                upload_folder,
                file
            )

            if os.path.isfile(file_path):
                os.remove(file_path)

        # Save uploaded PDFs
        for file in uploaded_files:

            save_path = os.path.join(
                upload_folder,
                file.name
            )

            with open(save_path, "wb") as f:
                f.write(file.getbuffer())

        with st.spinner(
            "Processing PDFs..."
        ):

            documents = process_pdfs(
                upload_folder
            )

            index = create_vector_store(
                documents
            )

            st.session_state.documents = documents
            st.session_state.index = index

        st.sidebar.success(
            f"Loaded {len(documents)} chunks"
        )

# -----------------------------
# Show Loaded PDFs
# -----------------------------

if os.path.exists("data/uploads"):

    files = os.listdir("data/uploads")

    if files:

        st.sidebar.subheader(
            "Loaded PDFs"
        )

        for file in files:
            st.sidebar.write(file)

# -----------------------------
# Chat History
# -----------------------------

for message in st.session_state.messages:

    with st.chat_message(
        message["role"]
    ):

        st.markdown(
            message["content"]
        )

# -----------------------------
# Chat Input
# -----------------------------

question = st.chat_input(
    "Ask a question about your PDFs..."
)

# -----------------------------
# Answer Questions
# -----------------------------

if question:

    if st.session_state.documents is None:

        st.warning(
            "Upload and process PDFs first."
        )

    else:

        st.session_state.messages.append(
            {
                "role": "user",
                "content": question
            }
        )

        with st.chat_message("user"):

            st.markdown(question)

        with st.spinner("Thinking..."):

            answer, sources = ask_rag(
                question,
                st.session_state.documents,
                st.session_state.index,
                st.session_state.llm
            )

        with st.chat_message("assistant"):

            st.markdown(answer)

            st.markdown("### Sources")

            for source in sources:

                st.markdown(
                    f"- {source}"
                )

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer
            }
        )
