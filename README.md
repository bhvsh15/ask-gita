#  Ask-Gita — RAG Backend

A lightweight Retrieval-Augmented-Generation (RAG) backend that answers questions from the **Bhagavad Gita** using:

- **ChromaDB** (persistent vector storage)
- **Ollama embeddings** (`nomic-embed-text`)
- **Semantic chunking** with LangChain
- **LLM querying** using **Groq** or **OpenRouter**
- Modular architecture with clean separation:
  - `build_index.py` → builds vector DB
  - `query_gita.py` → answers user questions

---

##  Features

- Fast local embeddings (Ollama)
- High-quality semantic chunking
- Persistent vectorstore using Chroma
- Use either:
  - **Groq (Llama 3.1 / Mixtral / Gemma)**
  - **OpenRouter**
- Simple 2-file backend deployment

---

## 📁 Project Structure

