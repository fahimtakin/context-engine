"""Builds the LangGraph state machine and its Redis-backed checkpointer."""

import logging
from collections.abc import AsyncIterator
from contextlib import AsyncExitStack, asynccontextmanager

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from app.agents.nodes import (
    decide_after_grading,
    generation_node,
    grade_documents_node,
    retrieve_node,
    rewrite_query_node,
    route_after_router,
    router_node,
    web_search_node,
)
from app.agents.state import AgentState
from app.core.config import settings

logger = logging.getLogger(__name__)


def build_workflow() -> StateGraph:
    workflow = StateGraph(AgentState)

    workflow.add_node("router", router_node)
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("grade", grade_documents_node)
    workflow.add_node("rewrite", rewrite_query_node)
    workflow.add_node("web_search", web_search_node)
    workflow.add_node("generate", generation_node)

    workflow.add_edge(START, "router")
    workflow.add_conditional_edges(
        "router",
        route_after_router,
        {"vector_db": "retrieve", "web_search": "web_search", "direct": "generate"},
    )
    workflow.add_edge("retrieve", "grade")
    workflow.add_conditional_edges(
        "grade",
        decide_after_grading,
        {"generate": "generate", "rewrite": "rewrite", "web_search": "web_search"},
    )
    workflow.add_edge("rewrite", "retrieve")
    workflow.add_edge("web_search", "generate")
    workflow.add_edge("generate", END)
    return workflow


def compile_agent(checkpointer: BaseCheckpointSaver | None = None):
    return build_workflow().compile(checkpointer=checkpointer)


@asynccontextmanager
async def open_checkpointer() -> AsyncIterator[BaseCheckpointSaver]:
    """Yield a Redis checkpointer, or an in-memory one if Redis is unreachable.

    The in-memory fallback keeps the service usable without Redis, at the cost of
    losing conversation memory on restart.
    """
    async with AsyncExitStack() as stack:
        try:
            from langgraph.checkpoint.redis.aio import AsyncRedisSaver

            saver = await stack.enter_async_context(
                AsyncRedisSaver.from_conn_string(settings.redis_url)
            )
            await saver.asetup()
            logger.info("Using Redis checkpointer at %s", settings.redis_url)
        except Exception:
            logger.warning(
                "Redis unavailable at %s; using in-memory checkpointer "
                "(conversation memory will not survive restarts)",
                settings.redis_url,
                exc_info=True,
            )
            saver = MemorySaver()
        yield saver
