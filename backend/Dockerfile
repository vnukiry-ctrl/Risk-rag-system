# Milestone 6.3 (Docker containerization). Doesn't containerize Qdrant as a
# separate service -- vector_store.py deliberately runs it in embedded
# on-disk mode (QdrantClient(path=...), see that file's own comment), not
# client-server mode, so there's no separate Qdrant process to containerize.
# Ollama likewise stays a host-level dependency (see docker-compose.yml's
# OLLAMA_BASE_URL, pointed at host.docker.internal) rather than a service
# here -- it's a GPU-benefiting daemon most dev setups already run outside
# any container, same posture as the README's existing setup instructions.
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
