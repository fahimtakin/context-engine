"""FastAPI gateway for ContextEngine."""

import logging
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field

from app.agents.graph import compile_agent, open_checkpointer
from app.core.config import settings
from app.core.llm import resolve_ollama_endpoint
from app.core.logging_setup import configure_logging
from app.db.qdrant_client import get_qdrant_client
from app.services.git_ingestion import ingest_github_repository
from app.services.ingestion import IngestionError

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- schemas


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=8000, description="The user's question")
    thread_id: str | None = Field(None, description="Session id; reuse it to keep memory")


class ChatResponse(BaseModel):
    response: str
    thread_id: str
    router_decision: str
    sources: list[str] = []
    retries: int = 0
    used_web_fallback: bool = False


class GitIndexRequest(BaseModel):
    repo_url: str = Field(..., description="Public https URL of a Git repository")
    branch: str = Field("main", description="Branch to index")


class GitIndexResponse(BaseModel):
    status: str
    message: str
    files: int
    chunks: int


# --------------------------------------------------------------------------- app factory


def create_app(agent: Any | None = None) -> FastAPI:
    """Build the API.

    Pass `agent` to inject a pre-built graph (used by tests); otherwise one is compiled
    at startup against a Redis checkpointer.
    """
    configure_logging(settings.LOG_LEVEL)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        if agent is not None:
            app.state.agent = agent
            app.state.checkpointer = "injected"
            yield
            return
        async with open_checkpointer() as saver:
            # Compile once at startup instead of on every request.
            app.state.agent = compile_agent(saver)
            app.state.checkpointer = type(saver).__name__
            logger.info("%s API ready", settings.PROJECT_NAME)
            yield

    app = FastAPI(title=f"{settings.PROJECT_NAME} API", version="1.1.0", lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ORIGINS,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    @app.get("/health")
    async def health() -> dict:
        """Liveness: the process is up."""
        return {"status": "ok"}

    @app.get("/ready")
    async def ready(request: Request) -> JSONResponse:
        """Readiness: can we reach the services a chat request depends on?"""
        checks = {
            "qdrant": await run_in_threadpool(_qdrant_ok),
            "ollama": await _ollama_ok(),
            "checkpointer": request.app.state.checkpointer,
        }
        healthy = checks["qdrant"] and checks["ollama"]
        return JSONResponse(
            status_code=200 if healthy else 503,
            content={"status": "ready" if healthy else "degraded", "checks": checks},
        )

    @app.post("/api/chat", response_model=ChatResponse)
    async def chat(payload: ChatRequest, request: Request) -> ChatResponse:
        thread_id = payload.thread_id or str(uuid.uuid4())
        config = {"configurable": {"thread_id": thread_id}}
        inputs = {"messages": [HumanMessage(content=payload.message)]}
        try:
            state = await request.app.state.agent.ainvoke(inputs, config=config)
        except Exception as err:
            logger.exception("Agent run failed for thread %s", thread_id)
            raise HTTPException(status_code=500, detail="The agent failed to answer.") from err

        return ChatResponse(
            response=state.get("final_output", ""),
            thread_id=thread_id,
            router_decision=state.get("router_decision", "vector_db"),
            sources=state.get("sources", []),
            retries=state.get("retry_count", 0),
            used_web_fallback=state.get("used_web_fallback", False),
        )

    @app.post("/api/index-git", response_model=GitIndexResponse)
    async def index_git(payload: GitIndexRequest) -> GitIndexResponse:
        try:
            # Cloning and embedding block, so keep them off the event loop.
            result = await run_in_threadpool(
                ingest_github_repository, payload.repo_url, payload.branch
            )
        except IngestionError as err:
            raise HTTPException(status_code=400, detail=str(err)) from err
        except Exception as err:
            logger.exception("Indexing failed for %s", payload.repo_url)
            raise HTTPException(status_code=500, detail="Indexing failed.") from err

        return GitIndexResponse(
            status="success", message=result.message, files=result.files, chunks=result.chunks
        )

    return app


def _qdrant_ok() -> bool:
    try:
        get_qdrant_client().get_collections()
        return True
    except Exception:
        return False


async def _ollama_ok() -> bool:
    try:
        async with httpx.AsyncClient(timeout=3) as client:
            response = await client.get(f"{resolve_ollama_endpoint()}/api/tags")
        return response.status_code == 200
    except Exception:
        return False


app = create_app()
