# Multi-stage production Dockerfile for Intelligent Knowledge Assistant
FROM python:3.12-slim

# Prevent Python from writing .pyc files and enable unbuffered output
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install essential system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application source code and pre-built vectorstore
COPY src/ ./src/
COPY data/ ./data/
COPY RAG_documents/ ./RAG_documents/
COPY scripts/ ./scripts/
COPY tests/ ./tests/
COPY app.py .
COPY .env.example .

# Expose FastAPI (8000) and Streamlit (8501)
EXPOSE 8000 8501

# Default command starts FastAPI service
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
