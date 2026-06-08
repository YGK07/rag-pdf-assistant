# 📚 RAG PDF Assistant (Streamlit + LangChain + Groq)

## 🚀 Live App
👉 https://rag-pdf-assistant-ygk.streamlit.app/

---

## 📌 Project Overview

The **RAG PDF Assistant** is a Retrieval-Augmented Generation (RAG) chatbot that allows users to upload multiple PDF files and ask questions based on their content.

It extracts text from PDFs, splits it into chunks, converts them into embeddings, stores them in a FAISS vector database, retrieves relevant context, and generates answers using the Groq LLM (LLaMA 3).

This ensures responses are **grounded in the uploaded documents** rather than generic LLM knowledge.

---

## ⚙️ Features

- 📄 Upload multiple PDF documents
- 🧠 AI-powered question answering (RAG pipeline)
- 🔍 Semantic search using embeddings (Sentence Transformers)
- 📚 Source tracking for every answer
- 💬 Chat-style interface (like ChatGPT)
- 🧾 Conversation history (session-based)
- ⚡ Fast responses using Groq API (LLaMA 3)
- 🗂️ Sidebar showing uploaded PDFs

---

## 🧠 Tech Stack

- Python
- Streamlit
- LangChain
- FAISS (Vector Database)
- Sentence Transformers
- Groq API (LLaMA 3)
- PyPDF / pdfplumber

---

## 🏗️ Architecture

PDF Upload → Text Extraction → Chunking → Embeddings → FAISS Vector Store → Retrieval → Groq LLM → Final Answer

---
rag-pdf-assistant/
│
├── app.py
├── rag.py
├── requirements.txt
│
├── data/
│ └── uploads/
│
├── vector_store/
└── README.md


---

## 🚀 How It Works

1. Upload PDFs via sidebar
2. PDFs are stored locally in `data/uploads/`
3. Text is extracted and split into chunks
4. Embeddings are created using Sentence Transformers
5. FAISS stores vector embeddings
6. User asks a question
7. Relevant chunks are retrieved
8. Groq LLM generates final answer

---
📊 What I Learned
Building a full RAG pipeline from scratch
Vector embeddings and semantic search
LangChain integration with LLMs
Deploying ML apps using Streamlit Cloud
Handling real-world dependency issues

👨‍💻 Author

Yohan George
B.Tech CSE (AI & ML), VIT Vellore



## 📁 Project Structure
