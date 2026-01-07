from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# ----------------------
# FastAPI setup
# ----------------------
app = FastAPI(title="RAG Chatbot")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------
# Root route (IMPORTANT)
# ----------------------
@app.get("/")
def root():
    return {"status": "RAG chatbot is running"}

# ----------------------
# Load resources ONCE
# ----------------------
embeddings = OllamaEmbeddings(model="nomic-embed-text")

VECTORSTORE = FAISS.load_local(
    "faiss_index",
    embeddings,
    allow_dangerous_deserialization=True
)

LLM = OllamaLLM(
    model="phi3",
    temperature=0,
    num_ctx=2048
)

# Warm-up
LLM.invoke("hello")

PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are a helpful assistant.
Answer ONLY using the context below.

Context:
{context}

Question:
{question}

Answer:
"""
)

CHAIN = PROMPT | LLM | StrOutputParser()

# ----------------------
# API schema
# ----------------------
class ChatRequest(BaseModel):
    question: str


class ChatResponse(BaseModel):
    answer: str


# ----------------------
# Chat endpoint
# ----------------------
@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    docs = VECTORSTORE.similarity_search(req.question, k=5,fetch_k=8)

    context = "\n\n".join(
        doc.page_content[:500] for doc in docs
    )

    answer = CHAIN.invoke({
        "context": context,
        "question": req.question
    })

    return {"answer": answer}
