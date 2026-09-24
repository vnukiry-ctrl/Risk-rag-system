# Milestone 6.3 (Docker containerization). Doesn't containerize Qdrant as a
# separate service -- vector_store.py deliberately runs it in embedded
# on-disk mode (QdrantClient(path=...), see that file's own comment), not
# client-server mode, so there's no separate Qdrant process to containerize.
# Embeddings are Voyage AI (hosted API, Milestone 6.4) rather than a local
# Ollama dependency, so there's no embeddings daemon to run here either.
FROM python:3.12-slim

# tesseract-ocr is a real, previously-documented requirement (README/
# Milestone 5 known gap): pip install pytesseract only installs the Python
# wrapper, never the OCR binary itself -- every environment this deploys to
# needs its own install, this image included.
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY *.py ./
COPY tests/ ./tests/

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
