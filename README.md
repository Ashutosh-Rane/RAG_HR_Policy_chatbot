# HR Compass – RAG-Based HR Policy Chatbot

HR Compass is a Retrieval-Augmented Generation (RAG) chatbot that enables
accurate, document-grounded question answering over HR policy documents.

## 🚀 Features
- PDF ingestion and semantic chunking
- FAISS vector database for retrieval
- Local LLM inference using Ollama
- FastAPI backend
- Web-based chatbot UI
- Hallucination-free, context-grounded responses

## 🛠 Tech Stack
- Python
- FastAPI
- FAISS
- Ollama (LLMs)
- Vector Embeddings
- HTML, CSS, JavaScript

## 📂 Project Structure
data #HR Policy PDF

app.py # FastAPI backend

data_loader.py # PDF ingestion & vector creation

retrieval.py # Semantic retrieval

generation.py # RAG logic

index.html # Web chatbot UI


## ▶️ How to Run

```bash
pip install -r requirements.txt
ollama pull phi3
ollama serve
uvicorn app:app --reload
Open index.html in your browser.
