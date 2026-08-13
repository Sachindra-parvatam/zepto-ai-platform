"""
main.py — FastAPI wrapper for the Zepto Support Assistant.

Run with:
    uvicorn main:app --host 0.0.0.0 --port 7860 --reload

POST /ask
  Request:  {"query": "<customer question>"}
  Response: {"answer": "...", "sources": [...], "confidence": 0.0–1.0}
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from graph import ask, ZeptoResponse

app = FastAPI(
    title="Zepto Support Assistant",
    description=(
        "A RAG-based customer support assistant grounded in Zepto's policy documents. "
        "Powered by LangGraph + ChromaDB + sentence-transformers. "
        "MOCK_LLM=1 (default) — deterministic, offline mode."
    ),
    version="1.0.0"
)


class QueryRequest(BaseModel):
    query: str


class QueryResponse(BaseModel):
    answer: str
    sources: list[str]
    confidence: float


@app.get("/")
def root():
    return {"message": "Zepto Support Assistant is running. POST to /ask"}


@app.post("/ask", response_model=QueryResponse)
def ask_endpoint(request: QueryRequest) -> QueryResponse:
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query must not be empty.")
    result: ZeptoResponse = ask(request.query)
    return QueryResponse(
        answer=result.answer,
        sources=result.sources,
        confidence=result.confidence
    )
