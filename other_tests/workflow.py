import os
import json
import time
import gradio as gr
from typing import List, TypedDict
from dotenv import load_dotenv

# LangChain Imports
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langgraph.graph import START, END, StateGraph

# 1. Load Environment Variables (for Claude API Key)
load_dotenv()

# --------------------------
# 2. Setup Models & DB
# --------------------------
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

vector_store = Chroma(
    collection_name="tire_docs",
    persist_directory="./chroma_tires",
    embedding_function=embedding_model
)

# Define LLMs globally so we don't re-init them every request
llm_mistral = ChatOllama(model="mistral", temperature=0)

# Only init Claude if the key exists, otherwise warn
if os.getenv("ANTHROPIC_API_KEY"):
    llm_claude = ChatAnthropic(
        model="claude-haiku-4-5-20251001", 
        temperature=0
    )
else:
    print("⚠️ ANTHROPIC_API_KEY not found. Claude option will fail if selected.")
    llm_claude = None

# --------------------------
# 3. Define State
# --------------------------
class RAGState(TypedDict):
    question: str
    model_choice: str  # New field to track which model to use
    context: List[Document]
    answer: str
    log_data: dict

# --------------------------
# 4. Define Nodes
# --------------------------

def retrieve(state: RAGState):
    """Step 1: Fetch docs"""
    docs = vector_store.similarity_search(state["question"], k=4)
    return {"context": docs}

def generate(state: RAGState):
    """Step 2: Select Model & Generate"""
    print(f"🧠 Generating with: {state['model_choice']}")
    
    # Switch Logic
    if state["model_choice"] == "Claude Haiku (Cloud)":
        if not llm_claude:
            return {"answer": "❌ Error: ANTHROPIC_API_KEY not set in .env file."}
        active_llm = llm_claude
    else:
        active_llm = llm_mistral

    # Format context
    context_text = "\n\n".join([d.page_content for d in state["context"]])
    
    prompt = ChatPromptTemplate.from_template("""
    You are a tire warranty expert. Use the context below to answer the question.
    
    Context:
    {context}
    
    Question: 
    {question}
    """)
    
    chain = prompt | active_llm
    response = chain.invoke({"context": context_text, "question": state["question"]})
    
    return {"answer": response.content}

def log_interaction(state: RAGState):
    """Step 3: Capture Data"""
    log_entry = {
        "timestamp": time.time(),
        "model_used": state["model_choice"],
        "user_question": state["question"],
        "generated_answer": state["answer"],
        "retrieved_sources": [d.metadata.get("source", "unknown") for d in state["context"]]
    }
    
    with open("query_logs.jsonl", "a") as f:
        f.write(json.dumps(log_entry) + "\n")
        
    return {"log_data": log_entry}

# --------------------------
# 5. Build Graph
# --------------------------
workflow = StateGraph(RAGState)

workflow.add_node("retrieve", retrieve)
workflow.add_node("generate", generate)
workflow.add_node("log", log_interaction)

workflow.add_edge(START, "retrieve")
workflow.add_edge("retrieve", "generate")
workflow.add_edge("generate", "log")
workflow.add_edge("log", END)

app = workflow.compile()

# --------------------------
# 6. Gradio UI with Dropdown
# --------------------------
def ask_workflow(message, history, model_choice):
    # We pass the model_choice from the dropdown into the state
    inputs = {
        "question": message, 
        "model_choice": model_choice
    }
    result = app.invoke(inputs)
    return result["answer"]

# Create the interface with an extra sidebar option
demo = gr.ChatInterface(
    fn=ask_workflow,
    title="🛞 Hybrid Tire RAG (Mistral & Claude)",
    description="Select your model backend below.",
    type="messages",
    additional_inputs=[
        gr.Dropdown(
            choices=["Mistral (Local)", "Claude Haiku (Cloud)"], 
            value="Mistral (Local)", 
            label="LLM Backend"
        )
    ]
)

if __name__ == "__main__":
    print("🚀 Launching RAG...")
    demo.launch(server_name="0.0.0.0", server_port=7860)