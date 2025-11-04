import gradio as gr
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain

# --------------------------
# 1. Model & Embeddings
# --------------------------
llm = ChatOllama(model="mistral", temperature=0.3)
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
# 4. Build Retrieval Chain
# --------------------------
document_chain = create_stuff_documents_chain(llm, prompt)
retrieval_chain = create_retrieval_chain(retriever, document_chain)

# --------------------------
# 5. Response Function
# --------------------------
def ask_tire_bot(message, history):
    try:
        result = retrieval_chain.invoke({"input": message})
        answer = result["answer"]
    except Exception as e:
        answer = f"⚠️ Error: {e}"

    # messages format for Gradio
    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": answer})
    return "", history

# --------------------------
# 6. Gradio Interface
# --------------------------
with gr.Blocks(title="Tire & Warranty Assistant") as demo:
    gr.Markdown("## 🛞 Tire & Warranty Assistant\nAsk about warranty coverage, maintenance, or tire safety.")
    chatbot = gr.Chatbot(label="Conversation", type="messages", height=500)
    msg = gr.Textbox(placeholder="Ask a question...", label="Your Question")

    clear = gr.Button("Clear Chat")

    state = gr.State([])

    msg.submit(ask_tire_bot, [msg, state], [msg, chatbot])
    clear.click(lambda: ([], ""), None, [chatbot, msg])

# --------------------------
# 7. Launch App
# --------------------------
if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
