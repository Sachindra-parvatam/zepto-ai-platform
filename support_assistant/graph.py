"""
graph.py — LangGraph StateGraph with 3 nodes:
  1. classify_intent
  2. retrieve_and_answer
  3. direct_answer

MOCK_LLM behaviour (default / unset or =1):
  - classify_intent: keyword heuristic, no LLM call
  - retrieve_and_answer: returns canned template from top retrieved chunk
  - direct_answer: returns fixed canned string

Optional MOCK_LLM=0 extension: calls real LLM (Groq) — not required for grading.
"""

import os
from typing import TypedDict, Annotated, Optional
from pydantic import BaseModel
import chromadb
from sentence_transformers import SentenceTransformer
from langgraph.graph import StateGraph, END

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
MOCK_LLM = os.environ.get("MOCK_LLM", "1") != "0"   # default: mock mode
CHROMA_DIR = os.path.join(os.path.dirname(__file__), "chroma_store")
COLLECTION_NAME = "zepto_policies"
EMBED_MODEL = "all-MiniLM-L6-v2"
TOP_K = 3

# Keywords for intent classification (mock mode)
POLICY_KEYWORDS = [
    "delivery", "return", "refund", "membership", "tracking",
    "cancel", "gift card", "support hours"
]

# ─────────────────────────────────────────────
# STRUCTURED OUTPUT SCHEMA
# ─────────────────────────────────────────────
class ZeptoResponse(BaseModel):
    answer: str
    sources: list[str]        # chunk/document IDs
    confidence: float         # 0.0–1.0

# ─────────────────────────────────────────────
# GRAPH STATE
# ─────────────────────────────────────────────
class GraphState(TypedDict):
    query: str
    intent: str                   # "policy_question" | "general_question"
    retrieved_chunks: list[dict]  # [{id, content, source}]
    response: Optional[ZeptoResponse]

# ─────────────────────────────────────────────
# LAZY SINGLETONS (loaded once on first use)
# ─────────────────────────────────────────────
_embed_model = None
_chroma_collection = None

def get_embed_model():
    global _embed_model
    if _embed_model is None:
        _embed_model = SentenceTransformer(EMBED_MODEL)
    return _embed_model

def get_collection():
    global _chroma_collection
    if _chroma_collection is None:
        client = chromadb.PersistentClient(path=CHROMA_DIR)
        _chroma_collection = client.get_collection(COLLECTION_NAME)
    return _chroma_collection

# ─────────────────────────────────────────────
# STRUCTURED PROMPT TEMPLATE (Task 2)
# Used by MOCK_LLM=0 extension; included here as text for completeness.
# ─────────────────────────────────────────────
SYSTEM_PROMPT_TEMPLATE = """
## Role
You are ZeptoBot, a helpful and accurate customer support assistant for Zepto, 
an Indian quick-commerce grocery delivery service.

## Context
You have been given a set of retrieved policy documents from Zepto's internal 
knowledge base as context. Each document is labelled with its source ID.

Retrieved context:
{retrieved_context}

## Task
Answer the customer's question strictly based on the retrieved context above.
Provide a clear, concise, and helpful answer.

## Format
Respond in plain prose. Do not use bullet points or numbered lists unless the 
policy itself is a list. Keep your answer under 150 words.

## Constraints
- Do NOT answer using information not present in the provided context.
- Do NOT speculate, invent policies, or reference outside sources.
- If the retrieved context does not contain enough information to answer the 
  question, say: "I'm sorry, I don't have that information available."

## Few-shot Example
Customer question: "How long does delivery take?"
Expected answer: "Zepto delivers within 10 to 30 minutes of order confirmation, 
depending on your delivery zone and current order volume."

## Customer Question
{query}
"""

# ─────────────────────────────────────────────
# NODES
# ─────────────────────────────────────────────

def classify_intent(state: GraphState) -> GraphState:
    """
    Node 1: Classify the query as policy_question or general_question.
    Mock mode (default): keyword heuristic — no LLM call.
    Real LLM mode (MOCK_LLM=0): calls Groq LLM.
    """
    query = state["query"].lower()

    if MOCK_LLM:
        # ── MOCK: keyword heuristic ──
        intent = "general_question"
        for kw in POLICY_KEYWORDS:
            if kw in query:
                intent = "policy_question"
                break
    else:
        # ── REAL LLM (optional extension) ──
        from groq import Groq
        client = Groq(api_key=os.environ["GROQ_API_KEY"])
        resp = client.chat.completions.create(
            model="llama3-8b-8192",
            messages=[{
                "role": "user",
                "content": (
                    f"Classify the following customer query as either "
                    f"'policy_question' or 'general_question'. "
                    f"Reply with only one of those two labels.\n\nQuery: {state['query']}"
                )
            }],
            max_tokens=10
        )
        raw = resp.choices[0].message.content.strip().lower()
        intent = "policy_question" if "policy" in raw else "general_question"

    return {**state, "intent": intent}


def retrieve_and_answer(state: GraphState) -> GraphState:
    """
    Node 2: For policy_question — embed query, retrieve top-K chunks,
    then generate an answer.
    Retrieval always runs for real (no API needed).
    Answer generation branches on MOCK_LLM.
    """
    query = state["query"]

    # ── RETRIEVAL (always real) ──
    model = get_embed_model()
    collection = get_collection()
    query_embedding = model.encode(query).tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=TOP_K,
        include=["documents", "metadatas", "distances"]
    )

    retrieved = []
    for i in range(len(results["ids"][0])):
        retrieved.append({
            "id": results["ids"][0][i],
            "content": results["documents"][0][i],
            "source": results["metadatas"][0][i].get("source", ""),
            "distance": results["distances"][0][i]
        })

    top_chunk = retrieved[0] if retrieved else {"content": "", "id": "none"}
    top_snippet = top_chunk["content"][:200]

    if MOCK_LLM:
        # ── MOCK: canned template answer ──
        answer = f"Based on the retrieved context: {top_snippet}"
        sources = [c["id"] for c in retrieved]
        confidence = 1.0

    else:
        # ── REAL LLM (optional extension) ──
        from groq import Groq
        import json

        client = Groq(api_key=os.environ["GROQ_API_KEY"])

        ctx_text = "\n\n".join(
            f"[{c['id']}] {c['content']}" for c in retrieved
        )
        prompt = SYSTEM_PROMPT_TEMPLATE.format(
            retrieved_context=ctx_text, query=query
        )

        answer = None
        sources = [c["id"] for c in retrieved]
        confidence = 0.9

        for attempt in range(3):
            try:
                resp = client.chat.completions.create(
                    model="llama3-8b-8192",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=200
                )
                answer = resp.choices[0].message.content.strip()
                # Validate Pydantic (will raise if invalid; not applicable for plain text)
                ZeptoResponse(answer=answer, sources=sources, confidence=confidence)
                break
            except Exception as e:
                if attempt < 2:
                    prompt += f"\n\nPrevious attempt failed validation: {e}. Please retry with a valid response."
                else:
                    answer = "Error: could not generate a valid response after 3 attempts."

    response = ZeptoResponse(
        answer=answer,
        sources=sources,
        confidence=confidence
    )
    return {**state, "retrieved_chunks": retrieved, "response": response}


def direct_answer(state: GraphState) -> GraphState:
    """
    Node 3: For general_question — no retrieval.
    Mock mode: fixed canned string.
    Real LLM mode (optional): direct LLM call.
    """
    if MOCK_LLM:
        answer = "I can only answer questions about Zepto policies right now."
        sources = []
        confidence = 1.0
    else:
        from groq import Groq
        client = Groq(api_key=os.environ["GROQ_API_KEY"])
        resp = client.chat.completions.create(
            model="llama3-8b-8192",
            messages=[{
                "role": "system",
                "content": "You are ZeptoBot, a customer support assistant. Answer helpfully."
            }, {
                "role": "user",
                "content": state["query"]
            }],
            max_tokens=200
        )
        answer = resp.choices[0].message.content.strip()
        sources = []
        confidence = 0.7

    response = ZeptoResponse(answer=answer, sources=sources, confidence=confidence)
    return {**state, "response": response}


# ─────────────────────────────────────────────
# ROUTING LOGIC (conditional edge)
# Does NOT depend on MOCK_LLM — only intent classification result
# ─────────────────────────────────────────────

def route_intent(state: GraphState) -> str:
    """Conditional edge: routes to retrieve_and_answer or direct_answer."""
    if state["intent"] == "policy_question":
        return "retrieve_and_answer"
    return "direct_answer"


# ─────────────────────────────────────────────
# BUILD GRAPH
# ─────────────────────────────────────────────

def build_graph():
    builder = StateGraph(GraphState)

    builder.add_node("classify_intent", classify_intent)
    builder.add_node("retrieve_and_answer", retrieve_and_answer)
    builder.add_node("direct_answer", direct_answer)

    builder.set_entry_point("classify_intent")

    builder.add_conditional_edges(
        "classify_intent",
        route_intent,
        {
            "retrieve_and_answer": "retrieve_and_answer",
            "direct_answer": "direct_answer"
        }
    )

    builder.add_edge("retrieve_and_answer", END)
    builder.add_edge("direct_answer", END)

    return builder.compile()


# Module-level compiled graph (used by FastAPI)
graph = build_graph()


def ask(query: str) -> ZeptoResponse:
    """Public entry point: run the graph for a given query string."""
    initial_state: GraphState = {
        "query": query,
        "intent": "",
        "retrieved_chunks": [],
        "response": None
    }
    final_state = graph.invoke(initial_state)
    return final_state["response"]
