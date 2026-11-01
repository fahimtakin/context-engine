"""Qdrant client, collection setup and vector-store access."""

import logging

from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams

from app.core.config import settings
from app.core.llm import get_embeddings

logger = logging.getLogger(__name__)


def get_qdrant_client() -> QdrantClient:
    return QdrantClient(url=settings.qdrant_url, timeout=30)


def init_qdrant_collection() -> None:
    client = get_qdrant_client()
    name = settings.QDRANT_COLLECTION
    if client.collection_exists(name):
        logger.info("Collection '%s' already exists", name)
        return
    client.create_collection(
        collection_name=name,
        vectors_config=VectorParams(size=settings.EMBEDDING_DIMENSIONS, distance=Distance.COSINE),
    )
    logger.info("Created collection '%s'", name)


def get_vector_store() -> QdrantVectorStore:
    return QdrantVectorStore.from_existing_collection(
        embedding=get_embeddings(),
        collection_name=settings.QDRANT_COLLECTION,
        url=settings.qdrant_url,
        timeout=30,
    )
