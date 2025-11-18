import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.embeddings import OllamaEmbeddings

from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

# -----------------------------
# 1. Configuration
# -----------------------------
PDF_PATH = "data/gita.pdf"
CHROMA_DIR = "data/chroma_index"
EMBED_MODEL = "nomic-embed-text"
HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# -----------------------------
# 2. Load PDF
# -----------------------------
print("📖 Loading PDF...")
loader = PyPDFLoader(PDF_PATH)
docs = loader.load()
print(f"✅ Loaded {len(docs)} pages")

# -----------------------------
# 3. Initialize Ollama Embeddings
# -----------------------------
print("🔍 Initializing Ollama Embeddings...")
embedding_model = OllamaEmbeddings(model=EMBED_MODEL)

# -----------------------------
# 4. Initial Character Split (to avoid long input)
# -----------------------------
print("🧩 Rough splitting before semantic chunking...")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
texts = text_splitter.split_documents(docs)
print(f"✅ Split into {len(texts)} manageable chunks")

# -----------------------------
# 5. Semantic Chunking with batching
# -----------------------------
semantic_splitter = SemanticChunker(
    embedding_model,
    breakpoint_threshold_type="percentile",
    breakpoint_threshold_amount=95.0
)

chunked_docs = []

for i, doc in enumerate(texts):
    try:
        sub_docs = semantic_splitter.create_documents([doc.page_content])
        chunked_docs.extend(sub_docs)
    except Exception as e:
        print(f"⚠️ Skipping chunk {i} due to error: {e}")

print(f"✅ Created {len(chunked_docs)} refined semantic chunks")

# -----------------------------
# 6. Create and Persist Vector Store
# -----------------------------
print("💾 Building Chroma vector database...")
vectordb = Chroma.from_documents(
    documents=chunked_docs,
    embedding=embedding_model,
    persist_directory=CHROMA_DIR
)
vectordb.persist()
print(f"✅ Index built and saved at: {CHROMA_DIR}")

# -----------------------------
# 7. Verify
# -----------------------------
query = "What is the meaning of karma yoga?"
results = vectordb.similarity_search(query, k=2)
for i, res in enumerate(results, 1):
    print(f"\n🔹 Result {i}:\n{res.page_content[:300]}...")
