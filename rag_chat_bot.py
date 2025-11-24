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

# 1. Setup Components
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_store = Chroma(
    collection_name="tire_docs",
    persist_directory="./chroma_tires",
    embedding_function=embedding_model
)
retriever = vector_store.as_retriever(search_kwargs={"k": 4})
llm = ChatAnthropic(model="claude-3-haiku-20240307", temperature=0)

# 2. Define Helper
def format_docs(docs):
    """Convert list of Documents to a single string for the Prompt"""
    return "\n\n".join(doc.page_content for doc in docs)

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

# 3. The "Chain" (Now returns a Dictionary, not just a string)
rag_chain = (
    # Step A: Retrieve docs and add them to the state as 'docs'
    RunnablePassthrough.assign(docs=itemgetter("question") | retriever)
    # Step B: Calculate the answer (using the docs we just grabbed)
    .assign(answer=(
        {
            "context_str": lambda x: format_docs(x["docs"]), # Turn docs into string for LLM
            "question": itemgetter("question"),
            "chat_history": itemgetter("chat_history")
        }
        | prompt
        | llm
        | StrOutputParser()
    ))
)

# 4. UI Function with Metadata Logging
def ask_claude(message, history):
    history_str = ""
    for turn in history:
        role = "User" if turn["role"] == "user" else "AI"
        content = turn["content"]
        history_str += f"{role}: {content}\n"

    # Run Chain
    result = rag_chain.invoke({
        "question": message, 
        "chat_history": history_str
    })
    
    # Extract sources
    source_files = [doc.metadata.get("source", "Unknown File") for doc in result["docs"]]

    # Logging
    log_entry = {
        "timestamp": time.time(),
        "question": message,
        "answer": result["answer"],
        "sources_used": source_files
    }
    
    with open("query_logs.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(log_entry) + "\n")
        
    return result["answer"]

# 5. Launch
demo = gr.ChatInterface(
    fn=ask_claude,
    title="Tire Bot",
    description="I log exactly which files I read to answer you.",
    type="messages"
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)