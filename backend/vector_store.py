from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
import os
from dotenv import load_dotenv

load_dotenv()

# ============================================================
# EMBEDDINGS PROVIDER - EASY SWAP HERE
# ============================================================
# Current: Ollama (Free, Open Source)
# To switch to Anthropic: See instructions below

EMBEDDINGS_PROVIDER = "ollama"  # Options: "ollama" or "anthropic"

if EMBEDDINGS_PROVIDER == "ollama":
    # ✓ CURRENT: Free, Open Source
    from langchain_community.embeddings import OllamaEmbeddings
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    print("Using Ollama embeddings (free, open-source)")

elif EMBEDDINGS_PROVIDER == "anthropic":
    # TODO: TO USE ANTHROPIC INSTEAD:
    # 1. pip install langchain-anthropic
    # 2. Set ANTHROPIC_API_KEY in .env
    # 3. Change EMBEDDINGS_PROVIDER = "anthropic"
    from langchain_anthropic import AnthropicEmbeddings
    embeddings = AnthropicEmbeddings(model="claude-3-5-sonnet-20241022")
    print("Using Anthropic embeddings")

def setup_vector_store():
    """Initialize Qdrant vector database"""
    
    # Create in-memory Qdrant (perfect for development)
    client = QdrantClient(":memory:")
    
    return client, embeddings

if __name__ == "__main__":
    print("Setting up vector store...")
    client, embeddings = setup_vector_store()
    print("✓ Vector store initialized successfully!")
    print(f"✓ Embeddings provider: {EMBEDDINGS_PROVIDER}")
    print("\nVector store ready to add documents!")