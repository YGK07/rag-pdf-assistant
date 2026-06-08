from rag import *

documents = process_pdfs(
    "data/uploads"
)

print(
    f"Loaded {len(documents)} chunks"
)

index = create_vector_store(
    documents
)

llm = load_llm()

question = "What is normalization?"

answer, sources = ask_rag(
    question,
    documents,
    index,
    llm
)

print("\nQuestion:")
print(question)

print("\nAnswer:")
print(answer)

print("\nSources:")
print(sources)