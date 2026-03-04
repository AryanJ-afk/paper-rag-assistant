# Research Paper RAG Assistant (Fully Local) — FastAPI + FAISS + Ollama + Docker

A **fully-local** Retrieval-Augmented Generation (RAG) assistant for research papers.
Upload PDFs, ask questions, generate structured summaries, and get **citation-grounded** answers — **no API keys** required.

---

## Key Features

### 1) Per-Paper Indexing (No Cross-Paper Contamination)
Each PDF gets its own FAISS index + metadata store, preventing retrieval from mixing content across papers.

### 2) Ask Questions (Q&A)
Query a specific paper for:
- main contributions
- methods / datasets / results
- limitations
- authors / affiliations (as present in the paper)

Answers include **page citations** like:
`[SomePaper.pdf p.12]`

### 3) Structured Summaries
Returns a consistent 5-part summary:
1) Problem  
2) Key idea / method  
3) Data / setup  
4) Results  
5) Limitations  

### 4) Prompt-Injection Resistant
Retrieved context is treated as **quoted text** and the model is instructed to **ignore instructions inside context**.

### 5) Evaluation (Retrieval Recall@K)
A lightweight retrieval evaluation script measures whether relevant evidence is retrieved.

### 6) Web UI
Simple homepage for:
- ingest PDF
- query paper
- summarize paper

### 7) Dockerized App
FastAPI app can run inside Docker for reproducible deployment.

---

## Tech Stack

| Component | Technology |
|---|---|
| API Server | FastAPI + Uvicorn |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` |
| Vector Search | FAISS |
| LLM | Ollama (`phi3:mini` / `llama3.1:8b`) |
| PDF Parsing | `pypdf` |
| UI | HTML + JS (FastAPI templates) |
| Evaluation | Custom Recall@K benchmark |
| Containerization | Docker |

