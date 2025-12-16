import json
import os
import smtplib
import pandas as pd
from datetime import datetime, timedelta
from typing import TypedDict, Annotated
from dotenv import load_dotenv
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# --- LangChain / LangGraph Imports ---
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_anthropic import ChatAnthropic
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langchain.agents import create_agent

load_dotenv()

LOG_FILE = "query_logs.jsonl"

# --- 1. Tool Setup ---
# Initialize the brain
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_store = Chroma(
    collection_name="tire_docs",
    persist_directory="./chroma_tires",
    embedding_function=embedding_model
)

@tool
def warranty_policy_search(query: str) -> str:
    """Useful for when you need to verify specific tire warranty numbers, mileage limits, or rotation rules."""
    
    # --- DEMO PRINT ---
    print(f"    🔎 [Tool Call] Searching Vector DB for: '{query}'...")
    
    docs = vector_store.similarity_search(query, k=3)
    return "\n\n".join([d.page_content for d in docs])

# --- 2. State Definition ---
class AgentState(TypedDict):
    raw_data: str
    stats_analysis: str
    confusion_analysis: str
    recommendations: str
    final_report: str
    email_status: str

# --- 3. Initialize LLM & Create Agents ---
llm = ChatAnthropic(model="claude-sonnet-4-5-20250929", temperature=0.3)

stats_agent = create_agent(llm, tools=[], name="Stats Agent")
confusion_agent = create_agent(llm, tools=[warranty_policy_search], name="Confusion Agent")
recs_agent = create_agent(llm, tools=[], name="Recommendations Agent")

# --- 4. Define Nodes ---

def load_data_node(state: AgentState):
    """Ingests logs and filters by date."""
    print("\n📂 [System] Loading and filtering log data...")
    days = 30 
    if not os.path.exists(LOG_FILE):
        return {"raw_data": "No log file found."}

    data = []
    with open(LOG_FILE, 'r') as f:
        for line in f:
            try:
                data.append(json.loads(line))
            except:
                continue
    
    df = pd.DataFrame(data)
    if 'timestamp' in df.columns:
        df['date'] = pd.to_datetime(df['timestamp'], unit='s')
        cutoff = datetime.now() - timedelta(days=days)
        df = df[df['date'] > cutoff]
    
    questions = df['question'].tolist() if 'question' in df.columns else []
    
    if not questions:
        print("   ⚠️ No recent questions found.")
        return {"raw_data": "No questions found in range."}

    print(f"   ✅ Loaded {len(questions)} recent questions.")
    return {"raw_data": "\n".join([f"- {q}" for q in questions])}

def stats_node(state: AgentState):
    """Worker 1: Extracts recurring topics."""
    print("📊 [Stats Agent] Analyzing recurring topics...")
    prompt = f"Analyze these user questions and list the Top 3 Recurring Topics with counts:\n\n{state['raw_data']}"
    result = stats_agent.invoke({"messages": [("user", prompt)]})
    return {"stats_analysis": result["messages"][-1].content}

def confusion_node(state: AgentState):
    """Worker 2: Investigator (Uses Tools)."""
    print("🕵️ [Confusion Agent] Investigating user queries (Checking Policies)...")
    prompt = (
        f"Analyze these user questions: {state['raw_data']}\n"
        "Explain WHY users are confused. If questions are about specific numbers (mileage, dates), "
        "use your search tool to check the official policy."
    )
    result = confusion_agent.invoke({"messages": [("user", prompt)]})
    return {"confusion_analysis": result["messages"][-1].content}

def recommendations_node(state: AgentState):
    """Worker 3: Training suggestions. NOW DEPENDS ON PREVIOUS ANALYSIS."""
    print("🎓 [Training Agent] Reading analysis and generating recommendations...")
    
    # Updated Prompt: Uses the output from Stats and Confusion
    prompt = f"""
    Based on the following analysis, suggest 3 concrete training topics.
    
    1. Recurring Topics Found: 
    {state['stats_analysis']}
    
    2. Areas of User Confusion:
    {state['confusion_analysis']}
    
    Original Data context:
    {state['raw_data']}
    """
    
    result = recs_agent.invoke({"messages": [("user", prompt)]})
    return {"recommendations": result["messages"][-1].content}

def email_compiler_node(state: AgentState):
    """Aggregator: Synthesizes the final report."""
    print("\n📧 [System] Compiling final executive report...")
    
    synthesis_prompt = f"""
    You are an automated AI Executive Reporting System.
    Synthesize the following distinct analyses into a single, cohesive and concise Executive Training Summary.
    
    DATA SOURCES:
    1. Recurring Topics: {state['stats_analysis']}
    2. Investigation Findings: {state['confusion_analysis']}
    3. Recommendations: {state['recommendations']}
    
    FORMATTING RULES:
    - Start with header "EXECUTIVE TRAINING SUMMARY".
    - Use clear sections with bold headers.
    - Remove conversational filler; keep it objective and concise.
    - End with "*** Report Generated by Automated AI Analytics Agent ***"
    """
    
    response = llm.invoke(synthesis_prompt)
    email_body = response.content
    
    # Execute Email Send
    sender = os.getenv("EMAIL_SENDER")
    password = os.getenv("EMAIL_PASSWORD")
    recipient = os.getenv("EMAIL_RECIPIENT")

    status = "Skipped (Config Missing)"
    if sender and password:
        try:
            msg = MIMEMultipart()
            msg['From'] = sender
            msg['To'] = recipient
            msg['Subject'] = f"Training Intelligence Report - {datetime.now().strftime('%B %Y')}"
            msg.attach(MIMEText(email_body, 'plain'))
            
            with smtplib.SMTP('smtp.gmail.com', 587) as server:
                server.starttls()
                server.login(sender, password)
                server.send_message(msg)
            status = "Sent Successfully"
        except Exception as e:
            status = f"Failed: {str(e)}"
            
    print(f"   ✅ Email Status: {status}")
    return {"final_report": email_body, "email_status": status}

# --- 5. Build Graph ---

builder = StateGraph(AgentState)

# Add Nodes
builder.add_node("load_data", load_data_node)
builder.add_node("stats", stats_node)
builder.add_node("confusion", confusion_node)
builder.add_node("recommendations", recommendations_node)
builder.add_node("email_compiler", email_compiler_node)

# --- Define Logic (UPDATED FLOW) ---
builder.add_edge(START, "load_data")

# Parallel: Load -> Stats AND Confusion
builder.add_edge("load_data", "stats")
builder.add_edge("load_data", "confusion")

# Sequential: Both Stats & Confusion must finish before Recommendations starts
builder.add_edge("stats", "recommendations")
builder.add_edge("confusion", "recommendations")

# Sequential: Recommendations -> Email
builder.add_edge("recommendations", "email_compiler")

builder.add_edge("email_compiler", END)

workflow = builder.compile()

# --- 6. Execution ---
if __name__ == "__main__":
    print("🚀 Starting Agentic Analytics Pipeline...")
    result = workflow.invoke({"raw_data": ""}) 
    
    print(f"\nSTATUS: {result['email_status']}")
    print("\n--- FINAL REPORT DRAFT ---\n")
    print(result['final_report'])