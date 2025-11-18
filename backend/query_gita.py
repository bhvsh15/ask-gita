import os
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.chat_models import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Load API keys
load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# --- Step 1: Embeddings & Vector Store ---
embedding_model = OllamaEmbeddings(model="nomic-embed-text",)
vectorstore = Chroma(persist_directory="data/chroma_index", embedding_function=embedding_model)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# --- Step 2: LLM (using OpenRouter via ChatOpenAI) ---
llm = ChatOpenAI(
    model_name="meta-llama/llama-3-70b-instruct",  
    openai_api_key=OPENROUTER_API_KEY,
    base_url="https://openrouter.ai/api/v1"
)

# --- Step 3: Prompt Template ---
prompt = PromptTemplate.from_template("""
You are an expert assistant answering questions based on the Bhagavad Gita.
Use the following context to answer the user's question concisely and accurately.

Context:
{context}

Question:
{question}

Answer:
""")

# --- Step 4: Format documents retrieved ---
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# --- Step 5: Define the RAG chain ---
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# --- Step 6: Query ---
if __name__ == "__main__":
    while True:
        query = input("\nAsk a question about the Gita (or type 'exit'): ")
        if query.lower() == "exit":
            break
        answer = rag_chain.invoke(query)
        print("\n🧠 Answer:", answer)
