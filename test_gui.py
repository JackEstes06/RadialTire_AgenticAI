import gradio as gr
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain

# --------------------------
# 1. Shared Setup
# --------------------------
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def build_retrieval_chain(model_name, temperature, k):
    """Create retrieval chain dynamically with adjustable params."""
    llm = ChatOllama(model=model_name, temperature=temperature)
    vectorstore = Chroma(
        collection_name="tire_docs",
        embedding_function=embedding_model,
        persist_directory="./chroma_tires"
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})

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

    document_chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(retriever, document_chain), retriever


# --------------------------
# 2. Main Response Function
# --------------------------
def ask_tire_bot(question, chat_history, model_name, temperature, k, show_context):
    """Handles user input and generates response."""
    if question.strip().lower() in ["exit", "quit"]:
        return "👋 Goodbye!", chat_history, ""

    # Build retrieval chain fresh each time to reflect settings
    retrieval_chain, retriever = build_retrieval_chain(model_name, temperature, k)

    # Perform retrieval and LLM reasoning
    result = retrieval_chain.invoke({"input": question})
    answer = result["answer"]

    # Retrieve and display the raw source context
    docs = retriever.invoke(question)

    context_snippets = "\n\n".join(
        [f"**Doc {i+1}:**\n{d.page_content[:600]}..." for i, d in enumerate(docs)]
    ) if show_context else ""

    chat_history.append({"role": "user", "content": "👷 " + question})
    chat_history.append({"role": "assistant", "content": "📘 " + answer})

    return "", chat_history, context_snippets


# --------------------------
# 3. Gradio UI Layout
# --------------------------
with gr.Blocks(title="Tire & Warranty Assistant") as demo:
    gr.Markdown("## 🛞 Tire & Warranty Assistant\nAsk about warranty coverage, maintenance, or tire safety.")

    with gr.Row():
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(height=500, label="Conversation", type="messages")
            context_box = gr.Markdown(label="Retrieved Context (for debugging)", visible=True)

        with gr.Column(scale=1):
            gr.Markdown("### ⚙️ Settings")

            model_name = gr.Dropdown(
                label="Model",
                choices=["phi3:mini", "llama3", "mistral", "phi3"],
                value="llama3"
            )

            temperature = gr.Slider(0.0, 1.0, value=0.3, step=0.05, label="Temperature")
            k = gr.Slider(1, 10, value=6, step=1, label="Top-K Documents")
            show_context = gr.Checkbox(value=True, label="Show retrieved context")

            question = gr.Textbox(
                placeholder="Ask a question about tires or warranties...",
                label="Employee Question"
            )

            submit = gr.Button("Ask")
            clear = gr.Button("Clear Chat")

    state = gr.State([])

    submit.click(
        ask_tire_bot,
        [question, state, model_name, temperature, k, show_context],
        [question, chatbot, context_box]
    )

    clear.click(lambda: ([], "", ""), None, [chatbot, question, context_box])

# --------------------------
# 4. Launch App
# --------------------------
if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
