from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import os
from insurance_loader import load_insurance_documents
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(
    title="Insurance RAG System",
    description="RAG system for insurance document analysis",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

groq_client = OpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1"
)

documents_db = {}


class QueryRequest(BaseModel):
    question: str
    top_k: int = 5


@app.get("/")
async def root():
    return {"status": "alive", "service": "Insurance RAG System"}


@app.post("/extract")
async def extract_documents():
    try:
        results = load_insurance_documents()
        extracted = []
        for meta in results["metadata"]:
            extracted.append(meta)
            doc_id = meta.get("policy_number") or meta.get("source_file", "unknown")
            documents_db[doc_id] = meta
        return {"total": len(extracted), "documents": extracted}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/documents")
async def list_documents():
    return {"total": len(documents_db), "documents": list(documents_db.values())}


@app.get("/health")
async def health_check():
    return {"status": "healthy", "documents_loaded": len(documents_db)}


@app.post("/search")
async def search_documents(query: str, top_k: int = 5):
    if not query:
        raise HTTPException(status_code=400, detail="query required")

    q = query.lower().strip()
    matches = []

    for key, doc in documents_db.items():
        text_blob = " ".join([
            str(doc.get("policy_number") or ""),
            str(doc.get("insurance_company") or ""),
            str(doc.get("insurance_type") or ""),
            str(doc.get("insured_name") or ""),
            str(doc.get("broker") or ""),
            str(doc.get("coverholder") or ""),
            str(doc.get("notes") or ""),
            " ".join(doc.get("key_coverages") or []),
            " ".join(doc.get("exclusions") or []),
        ]).lower()

        if q in text_blob:
            matches.append({
                "policy_number": doc.get("policy_number"),
                "insurance_company": doc.get("insurance_company"),
                "source": key,
                "preview": text_blob[:200],
            })

    return {"query": query, "total": len(matches[:top_k]), "results": matches[:top_k]}


@app.post("/query")
async def query_documents(request: QueryRequest):
    if not request.question:
        raise HTTPException(status_code=400, detail="question required")

    try:
        valid_docs = {k: v for k, v in documents_db.items() if "error" not in v}
        sources = list(valid_docs.keys())[:request.top_k]

        if valid_docs:
            context = "Context from insurance documents:\n\n"
            for doc_key in sources:
                doc = valid_docs[doc_key]
                context += f"=== {doc_key} ===\n"
                context += f"Company: {doc.get('insurance_company', 'N/A')}\n"
                context += f"Policy Type: {doc.get('insurance_type', 'N/A')}\n"
                context += f"Policy Number: {doc.get('policy_number', 'N/A')}\n"
                context += f"Insured: {doc.get('insured_name', 'N/A')}\n"
                context += f"Period: {doc.get('period_from', 'N/A')} to {doc.get('period_to', 'N/A')}\n"
                context += f"Premium: ${doc.get('premium_amount', 'N/A')}\n"
                context += f"Coverage Limit: {doc.get('coverage_limit', 'N/A')}\n"
                if doc.get('key_coverages'):
                    context += f"Coverages: {', '.join(doc.get('key_coverages', []))}\n"
                context += "\n"
        else:
            context = "No documents loaded. Please extract documents first."

        prompt = f"{context}\n\nUser Question: {request.question}\n\nProvide a clear, accurate answer based on the insurance documents above. If the answer is not in the documents, say so."

        response = groq_client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=[
                {"role": "system", "content": "You are an expert insurance policy analyst. Answer questions about insurance documents accurately and helpfully. If information is not in the provided documents, clearly state that."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=500
        )

        answer = response.choices[0].message.content

        return {
            "question": request.question,
            "answer": answer,
            "sources": sources,
            "documents_searched": len(documents_db)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)