import os
import json
import time
import gradio as gr
from operator import itemgetter
from dotenv import load_dotenv

# LangChain Imports
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

load_dotenv()

# --- 1. Setup Components ---
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

vector_store = Chroma(
    collection_name="tire_docs",
    persist_directory="./chroma_tires",
    embedding_function=embedding_model
)

# k=4 means we retrieve the 4 most relevant chunks per query
retriever = vector_store.as_retriever(search_kwargs={"k": 4})
llm = ChatAnthropic(model="claude-3-haiku-20240307", temperature=0)

# --- 2. Define Helper Functions ---
def format_docs(docs):
    """Joins retrieved document chunks into a single context string."""
    return "\n\n".join(doc.page_content for doc in docs)

# --- 3. Construct the RAG Chain ---
prompt = ChatPromptTemplate.from_template("""
You are a specialized Tire Warranty Assistant. Your job is to answer questions strictly based on the provided documentation.

Instructions:
1. Use ONLY the context below. Do not use outside knowledge.
2. If the answer is not in the context, say "I cannot find that information in the provided manuals."
3. If the context contains multiple tire models, specify which one you are referring to.
4. Keep answers concise and professional. Use bullet points if listing coverage details.

Conversation History:
{chat_history}

Context from manual:
{context_str}

Current Question: 
{question}

Answer:
""")

rag_chain = (
    # Step A: Retrieve docs and store them in the dictionary under 'docs'
    RunnablePassthrough.assign(docs=itemgetter("question") | retriever)
    # Step B: Pass retrieved docs + history + question to the LLM
    .assign(answer=(
        {
            "context_str": lambda x: format_docs(x["docs"]),
            "question": itemgetter("question"),
            "chat_history": itemgetter("chat_history")
        }
        | prompt
        | llm
        | StrOutputParser()
    ))
)

# --- 4. UI Function with Analytics Logging ---
def ask_claude(message, history):
    # Format chat history for the LLM
    history_str = ""
    for turn in history:
        role = "User" if turn["role"] == "user" else "AI"
        content = turn["content"]
        history_str += f"{role}: {content}\n"

    # Invoke the RAG Chain
    result = rag_chain.invoke({
        "question": message, 
        "chat_history": history_str
    })
    
    # Extract source metadata (filenames) for transparency
    source_files = [doc.metadata.get("source", "Unknown File") for doc in result["docs"]]

    # Log the interaction (This data powers Iteration 3)
    log_entry = {
        "timestamp": time.time(),
        "question": message,
        "answer": result["answer"],
        "sources_used": source_files
    }
    
    with open("query_logs.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(log_entry) + "\n")
        
    return result["answer"]

# --- 5. Launch Application ---
demo = gr.ChatInterface(
    fn=ask_claude,
    title="Tire Warranty Assistant (RAG)",
    description="I answer questions based strictly on uploaded PDF manuals. All queries are logged for analysis.",
    type="messages"
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)