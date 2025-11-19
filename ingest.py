import os
import shutil

from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# ----------------------
# Configuration
# ----------------------
PDF_DIR = "files"
CHROMA_PATH = "./chroma_tires"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

def main():
    # 1. Clean Legacy DB
    if os.path.exists(CHROMA_PATH):
        print(f"🧹 Clearing old database at {CHROMA_PATH}...")
        shutil.rmtree(CHROMA_PATH)

    # 2. Load Documents
    print(f"📂 Loading PDFs from '{PDF_DIR}'...")
    if not os.path.exists(PDF_DIR):
        os.makedirs(PDF_DIR)
        print(f"⚠️ Created missing '{PDF_DIR}' folder.")
        return

    loader = DirectoryLoader(
        PDF_DIR,
        glob="*.pdf",
        loader_cls=PyPDFLoader,
        show_progress=True,
        use_multithreading=True
    )
    docs = loader.load()
    
    if not docs:
        print("❌ No documents found.")
        return
        
    print(f"✅ Loaded {len(docs)} pages.")

    # 3. Split Text
    print("✂️  Splitting documents...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, 
        chunk_overlap=CHUNK_OVERLAP,
        add_start_index=True
    )
    chunks = text_splitter.split_documents(docs)
    print(f"🧩 Created {len(chunks)} text chunks.")

    # 4. Indexing
    print("💾 Indexing to ChromaDB...")
    
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    # This .from_documents method is the modern replacement for the manual loop
    Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=CHROMA_PATH,
        collection_name="tire_docs"
    )
    
    print("🚀 Success! Vector database is ready.")

if __name__ == "__main__":
    main()