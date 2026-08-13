# Module 3 — Support Assistant

## Overview

A complete RAG-based customer support assistant for Zepto, built with:
- **sentence-transformers** (`all-MiniLM-L6-v2`) for local embeddings
- **ChromaDB** for vector storage and cosine similarity retrieval
- **LangGraph** for the orchestration graph
- **FastAPI** for the REST API wrapper

**Default mode: `MOCK_LLM=1` (offline, deterministic — this is what gets graded).**  
No API key, no internet, no paid service required.

---

## Install & Run

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Build the ChromaDB index (run once)

```bash
cd support_assistant
python ingest.py
```

### 3. Start the FastAPI server

```bash
uvicorn main:app --host 0.0.0.0 --port 7860 --reload
```

### 4. Test the endpoint

```bash
# Policy question (triggers retrieval)
curl -X POST http://localhost:7860/ask \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the delivery fee?"}'

# General question (no retrieval)
curl -X POST http://localhost:7860/ask \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the capital of France?"}'
```

---

## Docker

```bash
# Build
docker build -t zepto-support .

# Run
docker run -p 7860:7860 zepto-support

# Test (same curl commands above)
```

The Dockerfile pre-downloads the sentence-transformer model and builds the ChromaDB index at image build time, so the container is fully self-contained.

---

## Example API Calls (MOCK_LLM=1 — graded baseline)

### Call 1 — Policy question (triggers retrieval)

**Request:**
```json
POST /ask
{"query": "What is the delivery fee for orders below INR 149?"}
```

**Response:**
```json
{
  "answer": "Based on the retrieved context: Zepto delivers grocery and household essentials to serviceable pin codes within 10 to 30 minutes of order confirmation, depending on the customer's delivery zone and current order volume. Standard delivery is free on orders over INR 149; orders below this threshold incur a flat INR 25 delivery fee.",
  "sources": ["doc_01_chunk_0", "doc_04_chunk_0", "doc_05_chunk_0"],
  "confidence": 1.0
}
```

### Call 2 — General question (no retrieval)

**Request:**
```json
POST /ask
{"query": "What is the weather like today?"}
```

**Response:**
```json
{
  "answer": "I can only answer questions about Zepto policies right now.",
  "sources": [],
  "confidence": 1.0
}
```

---

## RAG Pipeline Architecture

```
User Query
    │
    ▼
┌───────────────────┐
│  classify_intent  │  ← Node 1 (LangGraph)
│  (keyword heuristic│    Branches on MOCK_LLM toggle
│   in mock mode)   │
└─────────┬─────────┘
          │
    ┌─────┴──────┐
    │            │
    ▼            ▼
┌──────────┐  ┌─────────────┐
│  direct  │  │retrieve_and │  ← Node 2 / Node 3 (LangGraph)
│  answer  │  │   _answer   │
│ (Node 3) │  │  (Node 2)   │
└──────────┘  └──────┬──────┘
                     │
              ┌──────▼──────┐
              │  ChromaDB   │  ← cosine similarity, top-3 chunks
              │  retrieval  │    (always real, no API needed)
              └──────┬──────┘
                     │
              ┌──────▼──────┐
              │  Answer     │
              │ generation  │  ← MOCK: canned template from top chunk
              └──────┬──────┘    REAL: Groq LLM (optional)
                     │
              ┌──────▼──────┐
              │  Pydantic   │
              │  validation │  ← ZeptoResponse(answer, sources, confidence)
              └──────┬──────┘
                     │
                     ▼
                FastAPI /ask
                  response
```

### Stage-by-stage description

**Ingestion** (`ingest.py`)  
All 8 `.txt` policy documents in `docs/` are loaded, chunked (one chunk per document, since each is ~300 chars — well under the 500-char chunk size), embedded with `all-MiniLM-L6-v2`, and stored in a ChromaDB `PersistentClient` collection named `zepto_policies` under `chroma_store/`.

**Embedding** (`ingest.py` + `graph.py`)  
`SentenceTransformer("all-MiniLM-L6-v2")` runs entirely locally. No API call. Produces 384-dimensional vectors. Used at both index time (ingest) and query time (inside `retrieve_and_answer`).

**Retrieval** (`graph.py → retrieve_and_answer`)  
The query is embedded and ChromaDB's cosine similarity search returns the top-3 most similar chunks. This step **always runs for real** in both mock and real-LLM mode, since no network call is needed.

**Generation** (`graph.py → retrieve_and_answer / direct_answer`)  
This stage **branches on `MOCK_LLM`**:
- **Mock mode (default, graded)**: `retrieve_and_answer` returns `"Based on the retrieved context: <first 200 chars of top chunk>"`. `direct_answer` returns a fixed canned string. No LLM call.
- **Real LLM mode (`MOCK_LLM=0`, optional)**: Both nodes call the Groq API using the structured 5-part prompt template. `retrieve_and_answer` retries up to 2 additional times on Pydantic validation failure.

**Output schema** (`graph.py → ZeptoResponse`)  
Every response is validated against a Pydantic model:  
`{ answer: str, sources: list[str], confidence: float }`  
In mock mode, `sources` = chunk IDs of retrieved documents (empty for `direct_answer`), `confidence` = 1.0 (deterministic).

---

## Structured Prompt Template (Task 2 — used by MOCK_LLM=0 extension)

```
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
```

The 5 skeleton components present: **Role**, **Context**, **Task**, **Format**, **Constraints** (negative constraint: "Do NOT answer using information not present in the provided context"). One few-shot example is embedded.

---

## Design Decisions

- **One collection, one chunk per document**: Each document is small enough to be a single chunk. This keeps retrieval simple and ensures the top-3 results come from 3 different policy domains.
- **Cosine similarity**: Chosen over L2 distance because it is invariant to vector magnitude, making it better suited for semantic similarity of text embeddings.
- **Lazy singletons**: The embedding model and ChromaDB collection are loaded once on first use (not at import time), avoiding startup overhead and making testing easier.
- **`MOCK_LLM` toggle**: Gated at the top of `graph.py` with `os.environ.get("MOCK_LLM", "1") != "0"`, meaning the mock path is the default unless explicitly disabled.
- **Retry logic (MOCK_LLM=0 path)**: Up to 3 total attempts with a corrective instruction appended on failure, as required. In mock mode this code path is never reached.
