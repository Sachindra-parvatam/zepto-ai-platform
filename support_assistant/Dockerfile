# Dockerfile for Zepto Support Assistant FastAPI app
# Build:  docker build -t zepto-support .
# Run:    docker run -p 7860:7860 zepto-support
# Test:   curl -X POST http://localhost:7860/ask \
#           -H "Content-Type: application/json" \
#           -d '{"query": "What is the delivery fee?"}'

FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source
COPY . .

# Pre-download the sentence-transformer model at build time
# so the container works fully offline at runtime
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# Run ingestion to build the ChromaDB index at build time
RUN python ingest.py

# Expose the app port
EXPOSE 7860

# Default: MOCK_LLM=1 (offline, deterministic, no API key needed)
ENV MOCK_LLM=1

# Start FastAPI with uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
