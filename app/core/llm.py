"""Ollama clients for chat and embeddings, with host discovery when running in Docker."""

import logging
import os
import socket
from functools import lru_cache
from urllib.parse import urlparse

from langchain_ollama import ChatOllama, OllamaEmbeddings

from app.core.config import settings

logger = logging.getLogger(__name__)

# Addresses that commonly reach the Docker host from inside a container
# (Docker Desktop on Windows/macOS, and the default Linux bridge networks).
_HOST_GATEWAYS = ("host.docker.internal", "172.17.0.1", "172.18.0.1", "192.168.65.2", "10.0.75.2")


def running_in_container() -> bool:
    return os.path.exists("/.dockerenv") or os.environ.get("DOCKER_CONTAINER") == "true"


def _port_open(host: str, port: int, timeout: float = 0.5) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


@lru_cache(maxsize=1)
def resolve_ollama_endpoint() -> str:
    """Return a reachable Ollama base URL.

    Outside Docker this is simply OLLAMA_BASE_URL. Inside a container, Ollama usually
    runs on the host, so if the configured URL is unreachable we probe the usual host
    gateways and use the first one that answers. The result is cached for the process.
    """
    configured = settings.OLLAMA_BASE_URL
    if not running_in_container():
        return configured

    parsed = urlparse(configured)
    port = parsed.port or 11434
    if parsed.hostname and _port_open(parsed.hostname, port):
        return configured

    for gateway in _HOST_GATEWAYS:
        if _port_open(gateway, port):
            url = f"http://{gateway}:{port}"
            logger.info("Ollama not reachable at %s; using host gateway %s", configured, url)
            return url

    logger.warning("Could not reach Ollama from the container; falling back to %s", configured)
    return configured


def get_llm() -> ChatOllama:
    return ChatOllama(
        base_url=resolve_ollama_endpoint(),
        model=settings.LLM_MODEL,
        temperature=0,
        client_kwargs={"timeout": settings.LLM_TIMEOUT_SECONDS},
    )


def get_embeddings() -> OllamaEmbeddings:
    return OllamaEmbeddings(base_url=resolve_ollama_endpoint(), model=settings.EMBEDDING_MODEL)
