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

# k=8 means we retrieve the 8 most relevant chunks per query
retriever = vector_store.as_retriever(search_kwargs={"k": 8})
llm = ChatAnthropic(model="claude-sonnet-4-5-20250929", temperature=0)

# --- 2. Define Helper Functions ---
def format_docs(docs):
    """Joins retrieved document chunks into a single context string."""
    return "\n\n".join(doc.page_content for doc in docs)

# --- 3. Construct the RAG Chain ---
prompt = ChatPromptTemplate.from_template("""
You are a Senior Warranty Support Specialist for a tire shop. Your users are technicians who need accurate, internal-facing answers.

**INTENT DETECTION:**
- **IF** the user describes a specific scenario (e.g., "flat tire at 14 months", "22k miles on [Brand]"), treat this as a **CLAIM ADJUDICATION**.
- **IF** the user asks a general question (e.g., "What is the [Brand] road hazard policy?", "Does [Brand] cover spares?"), treat this as a **POLICY LOOKUP**.

**INSTRUCTIONS FOR CLAIM ADJUDICATION (Specific Scenarios):**
1.  **Verdict:** Determine if the claim is Valid, Invalid, or Conditional based *strictly* on that brand's manual.
2.  **Output Format:**
    * **STATUS:** [✅ Covered / ❌ Not Covered / ⚠️ Conditional]
    * **REASON:** 1-2 sentences explaining *why* based on the specific math/rules (e.g., "14 months exceeds the 12-month limit").
    * **PROOF:** Cite the specific limit from the text (e.g., "Policy Limit: 12 months or 2/32\" wear").
    * **NEXT STEPS:** What should the tech do next? (e.g., "Measure tread depth", "Deny claim").

**INSTRUCTIONS FOR POLICY LOOKUP (General Questions):**
1.  **Summary:** Briefly explain the coverage scope.
2.  **Output Format:**
    * **POLICY OVERVIEW:** A concise summary of the rule.
    * **CRITICAL LIMITS:** Bullet points for hard numbers (Mileage, Time, Tread Depth).
    * **EXCLUSIONS:** What voids this specific policy? (e.g., "Commercial use", "Repairable punctures").
    * **TECH NOTES:** Internal advice (e.g., "Always check DOT date first").

**DATA INTERPRETATION LOGIC (For Tables & Charts):**
- **Messy Tables:** If you see a stream of numbers (e.g., "21000 30 48 58 62"), interpret this as a **Mileage Adjustment Table**.
    - The first number is the **Mileage Driven**.
    - The subsequent numbers are **Credit Percentages** for different warranty tiers (e.g., 30k, 40k, 50k, 55k).
    - Map the user's tire model to its warranty tier to find the correct percentage.
- **Cross-Referencing:** Always search for a "Product List" or "Summary" to confirm which warranty tier a specific tire model belongs to.

**UNIVERSAL RULES (CRITICAL):**
- **BRAND ISOLATION PROTOCOL:**
    - **VERIFY SOURCE:** Before citing a rule, check if the source text belongs to the brand the user asked about.
    - **NO CROSS-CONTAMINATION:** Never apply a rule from one brand to another.
- **Accuracy:** Do not guess. If the manual doesn't say, say "Unknown/Check with Manager."
- **Tone:** Professional, direct, and internal-facing. No "customer service" fluff.

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