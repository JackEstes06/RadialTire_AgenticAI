from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain

# --------------------------
# 1. Model & Embeddings
# --------------------------
llm = ChatOllama(model="phi3", temperature=0.3)
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# --------------------------
# 2. Load Chroma Vectorstore
# --------------------------
vectorstore = Chroma(
    collection_name="tire_docs",
    embedding_function=embedding_model,
    persist_directory="./chroma_tires"
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 6})

# --------------------------
# 3. Prompt Template
# --------------------------
prompt = ChatPromptTemplate.from_template("""
You are a knowledgeable assistant specializing in tire information and warranty policies.

Use ONLY the provided context to answer the employee's question.
If the information is not in the context, say:
"The documentation does not specify that detail."

Context:
{context}

Employee question:
{input}

Answer clearly and concisely:
""")

# --------------------------
# 4. Build Retrieval Chain (modern way)
# --------------------------
document_chain = create_stuff_documents_chain(llm, prompt)
retrieval_chain = create_retrieval_chain(retriever, document_chain)

# --------------------------
# 5. Simple Interactive Loop
# --------------------------
print("🛞 Tire & Warranty Assistant")
print("Ask about warranty coverage, maintenance, or tire safety.")
print("Type 'exit' or 'quit' to end the conversation.\n")

while True:
    question = input("👷 Ask a question: ")
    if question.lower() in ["exit", "quit"]:
        print("👋 Goodbye")
        break

    result = retrieval_chain.invoke({"input": question})
    print("\n📘 Response:\n", result["answer"])
