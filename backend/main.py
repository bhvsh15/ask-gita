# backend/main.py
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from query_gita import rag_chain  

app = FastAPI()

# Allow frontend (adjust origin for production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    question: str

@app.post("/ask")
async def ask_gita(request: QueryRequest):
    try:
        question = request.question.strip()
        if not question:
            return {"error": "Question cannot be empty"}
        answer = rag_chain.invoke(question)
        return {"answer": answer}
    except Exception as e:
        return {"error": str(e)}


@app.get("/")
async def home():
    return {"message": "Ask Gita API is running 🙏"}
