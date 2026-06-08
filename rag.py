import os
import pdfplumber
import numpy as np
import faiss

from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os

load_dotenv()

embedding_model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
def extract_text_from_pdf(pdf_path):

    text = ""

    with pdfplumber.open(pdf_path) as pdf:

        for page in pdf.pages:

            page_text = page.extract_text()

            if page_text:

                text += page_text + "\n"

    return text

def process_pdfs(upload_folder):

    documents = []

    for filename in os.listdir(upload_folder):

        if filename.endswith(".pdf"):

            pdf_path = os.path.join(
                upload_folder,
                filename
            )

            text = extract_text_from_pdf(
                pdf_path
            )

            chunks = splitter.split_text(
                text
            )

            for chunk in chunks:

                documents.append({
                    "text": chunk,
                    "source": filename
                })

    return documents
def create_vector_store(documents):

    texts = [
        doc["text"]
        for doc in documents
    ]

    embeddings = embedding_model.encode(
        texts
    )

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(
        dimension
    )

    index.add(
        np.array(embeddings)
        .astype("float32")
    )

    return index
def retrieve_context(
    question,
    documents,
    index,
    k=3
):

    query_embedding = embedding_model.encode(
        [question]
    )

    distances, indices = index.search(
        np.array(query_embedding)
        .astype("float32"),
        k
    )

    retrieved_docs = []

    for idx in indices[0]:

        retrieved_docs.append(
            documents[idx]
        )

    return retrieved_docs
def load_llm():

    llm = ChatGroq(
        groq_api_key=os.getenv("GROQ_API_KEY"),
        model="llama-3.1-8b-instant",
        temperature=0
    )

    return llm
def ask_rag(
    question,
    documents,
    index,
    llm
):

    retrieved_docs = retrieve_context(
        question,
        documents,
        index
    )

    context = "\n\n".join(
        [
            doc["text"]
            for doc in retrieved_docs
        ]
    )

    sources = list(
        set(
            [
                doc["source"]
                for doc in retrieved_docs
            ]
        )
    )

    prompt = f"""
    You are a helpful study assistant.

    Use ONLY the context below.

    If the answer is not present,
    reply:

    I could not find that information
    in the provided documents.

    Context:
    {context}

    Question:
    {question}

    Answer:
    """

    response = llm.invoke(
        prompt
    )

    return (
        response.content,
        sources
    )
