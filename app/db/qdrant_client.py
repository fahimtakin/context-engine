# app/db/qdrant_client.py
import sys
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from app.core.config import settings

COLLECTION_NAME = "context_engine_docs"

def get_qdrant_client() -> QdrantClient:
    return QdrantClient(host=settings.QDRANT_HOST, port=settings.QDRANT_PORT)

def init_qdrant_collection():
    client = get_qdrant_client()
    collections = client.get_collections().collections
    exists = any(c.name == COLLECTION_NAME for c in collections)
    
    if not exists:
        # Routing messages to stderr ensures the MCP stdio pipe doesn't break!
        print(f"Creating collection: {COLLECTION_NAME}...", file=sys.stderr)
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=1024, distance=Distance.COSINE),
        )
        print("Collection created successfully!", file=sys.stderr)
    else:
        print(f"Collection '{COLLECTION_NAME}' already exists.", file=sys.stderr)
