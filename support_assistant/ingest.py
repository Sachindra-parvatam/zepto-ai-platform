"""
ingest.py — Load the 8 corpus documents, embed with all-MiniLM-L6-v2,
and store in a ChromaDB collection.
Run this once before starting the FastAPI app.
"""

import os
import chromadb
from sentence_transformers import SentenceTransformer

DOCS_DIR = os.path.join(os.path.dirname(__file__), "docs")
CHROMA_DIR = os.path.join(os.path.dirname(__file__), "chroma_store")
COLLECTION_NAME = "zepto_policies"
MODEL_NAME = "all-MiniLM-L6-v2"


def load_documents(docs_dir: str) -> list[dict]:
    """Load all .txt files from the docs directory."""
    docs = []
    for fname in sorted(os.listdir(docs_dir)):
        if fname.endswith(".txt"):
            fpath = os.path.join(docs_dir, fname)
            with open(fpath, "r", encoding="utf-8") as f:
                content = f.read().strip()
            doc_id = fname.replace(".txt", "")   # e.g. "doc_01"
            docs.append({"id": doc_id, "content": content, "source": fname})
    return docs


def chunk_document(doc: dict, chunk_size: int = 500) -> list[dict]:
    """
    Simple fixed-size character chunking.
    For documents this short (~300 chars each), one chunk per document is fine.
    We split on sentence boundaries if the doc exceeds chunk_size.
    """
    text = doc["content"]
    if len(text) <= chunk_size:
        return [{
            "id": doc["id"] + "_chunk_0",
            "content": text,
            "source": doc["source"],
            "doc_id": doc["id"]
        }]
    # Split into sentences and group into chunks
    sentences = text.split(". ")
    chunks = []
    current = ""
    idx = 0
    for sent in sentences:
        if len(current) + len(sent) < chunk_size:
            current += sent + ". "
        else:
            chunks.append({
                "id": f"{doc['id']}_chunk_{idx}",
                "content": current.strip(),
                "source": doc["source"],
                "doc_id": doc["id"]
            })
            current = sent + ". "
            idx += 1
    if current.strip():
        chunks.append({
            "id": f"{doc['id']}_chunk_{idx}",
            "content": current.strip(),
            "source": doc["source"],
            "doc_id": doc["id"]
        })
    return chunks


def build_index():
    """Embed all chunks and store in ChromaDB."""
    print(f"  Loading documents from: {DOCS_DIR}")
    raw_docs = load_documents(DOCS_DIR)
    print(f"  Loaded {len(raw_docs)} documents.")

    # Chunk
    all_chunks = []
    for doc in raw_docs:
        all_chunks.extend(chunk_document(doc))
    print(f"  Total chunks: {len(all_chunks)}")

    # Embed
    print(f"  Loading embedding model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)
    texts = [c["content"] for c in all_chunks]
    embeddings = model.encode(texts, show_progress_bar=True).tolist()
    print(f"  Embeddings shape: {len(embeddings)} × {len(embeddings[0])}")

    # Store in ChromaDB
    client = chromadb.PersistentClient(path=CHROMA_DIR)

    # Delete existing collection if re-running
    try:
        client.delete_collection(COLLECTION_NAME)
        print(f"  Deleted existing collection '{COLLECTION_NAME}'")
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )

    collection.add(
        ids=[c["id"] for c in all_chunks],
        embeddings=embeddings,
        documents=texts,
        metadatas=[{"source": c["source"], "doc_id": c["doc_id"]} for c in all_chunks]
    )
    print(f"  Indexed {collection.count()} chunks into ChromaDB collection '{COLLECTION_NAME}'.")
    print(f"  ChromaDB store path: {CHROMA_DIR}")
    return collection


if __name__ == "__main__":
    build_index()
    print("[✓] Ingestion complete.")
