from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb import PersistentClient
from chromadb.config import Settings
import os


# ----------------------
# 1. Setup
# ----------------------
PDF_DIR = "files"
CHROMA_PATH = "./chroma_tires"
COLLECTION_NAME = "tire_docs"
CHUNK_SIZE = 700
CHUNK_OVERLAP = 100

# ----------------------
# 2. Load PDFs
# ----------------------
def load_all_pdfs(folder_path):
    documents = []
    for fname in os.listdir(folder_path):
        if fname.lower().endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(folder_path, fname))
            pages = loader.load()
            for p in pages:
                p.metadata["source_file"] = fname
                documents.append(p)
    return documents

docs = load_all_pdfs(PDF_DIR)

# ----------------------
# 3. Chunk the text
# ----------------------
splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
chunks = splitter.split_documents(docs)

texts = [c.page_content for c in chunks]
metadatas = [c.metadata for c in chunks]

# ----------------------
# 4. Embed + Store
# ----------------------
model = SentenceTransformer("all-MiniLM-L6-v2")

client = PersistentClient(path=CHROMA_PATH, settings=Settings(anonymized_telemetry=False))
collection = client.get_or_create_collection(COLLECTION_NAME)

for i in range(0, len(texts), 200):
    batch_texts = texts[i:i+200]
    batch_metadatas = metadatas[i:i+200]
    embeddings = model.encode(batch_texts)
    collection.add(
        documents=batch_texts,
        embeddings=embeddings.tolist(),
        metadatas=batch_metadatas,
        ids=[f"chunk_{i+j}" for j in range(len(batch_texts))]
    )
    print(f"Processed {i+len(batch_texts)} / {len(texts)}")

print("✅ All tire warranty PDFs embedded successfully!")
