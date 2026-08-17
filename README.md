# Risk-RAG-System

A professional Retrieval Augmented Generation (RAG) system built with:
- **Backend**: FastAPI
- **Frontend**: Streamlit
- **RAG Framework**: LangChain
- **Vector Database**: Qdrant
- **LLM**: Claude (Anthropic API)

## Project Overview

A production-ready RAG system that demonstrates professional software architecture and best practices for building intelligent document retrieval systems. The system combines semantic search with large language models to provide accurate, source-attributed answers from custom knowledge bases.

## Features

- Multi-format document loading (PDF, DOCX, TXT)
- Intelligent document chunking and preprocessing
- Semantic search with embeddings
- LLM-based generation with source attribution
- Conversational chat interface
- Production-ready API architecture

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Backend API | FastAPI |
| Frontend UI | Streamlit |
| Document Processing | LangChain |
| Vector Storage | Qdrant |
| Embeddings | Anthropic Embeddings |
| LLM | Claude 3.5 Sonnet |

## Setup

### Backend
```bash
cd backend
python -m venv venv
venv\Scripts\activate
pip install langchain langchain-community langchain-anthropic qdrant-client fastapi uvicorn python-dotenv pypdf python-docx
```

### Frontend
```bash
cd frontend
python -m venv venv
venv\Scripts\activate
pip install streamlit requests
```

## Project Status

- [x] Step 1: Project structure
- [x] Step 2: Backend dependencies
- [x] Step 3: Frontend dependencies
- [x] Step 4: Document loader (PDF, DOCX, TXT)
- [ ] Step 5: Vector database setup
- [ ] Step 6: RAG chain implementation
- [ ] Step 7: FastAPI backend
- [ ] Step 8: Streamlit frontend
- [ ] Step 9: Testing & optimization
- [ ] Step 10: Deployment

## Author
Victoria Nukiry