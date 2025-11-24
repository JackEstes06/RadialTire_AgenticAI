import os
import shutil
import gradio as gr
from operator import itemgetter
from dotenv import load_dotenv

# LangChain Imports
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Import the new analytics module
import analytics 
import time
import json

load_dotenv()

CHROMA_PATH = "./chroma_tires"
PDF_STORAGE_PATH = "./uploaded_files"
os.makedirs(PDF_STORAGE_PATH, exist_ok=True)

# ---------------------------------------------------------
# GLOBAL SETUP
# ---------------------------------------------------------
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
retriever = None

def get_retriever():
    global retriever
    vector_store = Chroma(
        collection_name="tire_docs",
        persist_directory=CHROMA_PATH,
        embedding_function=embedding_model
    )
    try:
        if not vector_store.get()['ids']: return None
    except: return None
    retriever = vector_store.as_retriever(search_kwargs={"k": 4})
    return retriever

get_retriever()

# ---------------------------------------------------------
# TAB 1: CHAT LOGIC
# ---------------------------------------------------------
def ask_claude(message, history):
    if not retriever:
        return "⚠️ Database is empty! Upload manuals in the 'Manage Knowledge Base' tab."
        
    llm = ChatAnthropic(model="claude-3-haiku-20240307", temperature=0)
    prompt = ChatPromptTemplate.from_template("""
    You are a specialized Tire Warranty Assistant.
    Conversation History: {chat_history}
    Context: {context_str}
    Question: {question}
    Answer:
    """)

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
        RunnablePassthrough.assign(docs=itemgetter("question") | retriever)
        .assign(answer=(
            {
                "context_str": lambda x: format_docs(x["docs"]),
                "question": itemgetter("question"),
                "chat_history": itemgetter("chat_history")
            } | prompt | llm | StrOutputParser()
        ))
    )
    
    history_str = "\n".join([f"{'User' if t['role']=='user' else 'AI'}: {t['content']}" for t in history])
    result = rag_chain.invoke({"question": message, "chat_history": history_str})
    
    # Logging
    source_files = [doc.metadata.get("source", "Unknown") for doc in result["docs"]]
    log_entry = {
        "timestamp": time.time(),
        "question": message,
        "answer": result["answer"],
        "sources_used": source_files
    }
    with open("query_logs.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(log_entry) + "\n")
        
    return result["answer"]

# ---------------------------------------------------------
# TAB 2: INGESTION LOGIC
# ---------------------------------------------------------
def ingest_files(files):
    if not files: return "⚠️ No files."
    status = "🔄 Starting...\n"
    
    if os.path.exists(CHROMA_PATH): shutil.rmtree(CHROMA_PATH)
    if os.path.exists(PDF_STORAGE_PATH): shutil.rmtree(PDF_STORAGE_PATH)
    os.makedirs(PDF_STORAGE_PATH)
    
    saved_paths = []
    for f in files:
        name = os.path.basename(f.name)
        dest = os.path.join(PDF_STORAGE_PATH, name)
        shutil.copy(f.name, dest)
        saved_paths.append(dest)
        
    docs = []
    for p in saved_paths:
        docs.extend(PyPDFLoader(p).load())
        
    chunks = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200).split_documents(docs)
    Chroma.from_documents(chunks, embedding_model, persist_directory=CHROMA_PATH, collection_name="tire_docs")
    get_retriever()
    return status + f"✅ Success! Loaded {len(docs)} pages."

# ---------------------------------------------------------
# TAB 3: HR ANALYTICS LOGIC
# ---------------------------------------------------------
def run_analytics(hr_email):
    # 1. Generate Report
    report = analytics.get_monthly_analysis(days_back=30)
    
    # 2. Send Email (if address provided)
    email_status = "📧 No email address provided, report generated strictly for view."
    if hr_email and "@" in hr_email:
        email_status = analytics.send_hr_email(report, hr_email)
        
    return report, email_status

# ---------------------------------------------------------
# UI LAYOUT
# ---------------------------------------------------------
with gr.Blocks(title="Tire Agent & Admin") as demo:
    gr.Markdown("# 🛞 Tire Warranty AI System")
    
    with gr.Tabs():
        # TAB 1
        with gr.Tab("💬 Employee Chat"):
            gr.ChatInterface(fn=ask_claude, type="messages")
            
        # TAB 2
        with gr.Tab("📂 Manage Knowledge Base"):
            gr.Markdown("### Upload New Manuals")
            file_input = gr.File(file_count="multiple", file_types=[".pdf"])
            ingest_btn = gr.Button("Rebuild Database")
            ingest_log = gr.Textbox(label="Status")
            ingest_btn.click(ingest_files, file_input, ingest_log)
            
        # TAB 3
        with gr.Tab("📊 HR Training Insights"):
            gr.Markdown("### Monthly Training Report Generator")
            gr.Markdown("This tool analyzes employee search logs to find common knowledge gaps.")
            
            with gr.Row():
                hr_email_input = gr.Textbox(label="HR Manager Email (Optional)")
                analyze_btn = gr.Button("Generate Report & Email", variant="primary")
            
            status_output = gr.Textbox(label="Email Status")
            report_output = gr.Textbox(label="Generated Training Report", lines=15, interactive=False)
            
            analyze_btn.click(
                fn=run_analytics,
                inputs=hr_email_input,
                outputs=[report_output, status_output]
            )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)