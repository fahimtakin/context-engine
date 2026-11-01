"""MCP server exposing ContextEngine as tools for Claude Desktop (stdio transport).

All logging goes to stderr: stdout carries the JSON-RPC stream.
"""

import asyncio
import logging
import os
import sys
from contextlib import AsyncExitStack

# Make `app` importable and `.env` discoverable when launched as a plain script.
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

from langchain_core.messages import HumanMessage  # noqa: E402
from mcp.server.mcpserver import MCPServer  # noqa: E402

from app.agents.graph import compile_agent, open_checkpointer  # noqa: E402
from app.core.config import settings  # noqa: E402
from app.core.logging_setup import configure_logging  # noqa: E402
from app.services.git_ingestion import ingest_github_repository  # noqa: E402
from app.services.ingestion import IngestionError, ingest_directory  # noqa: E402

configure_logging(settings.LOG_LEVEL)
logger = logging.getLogger("context_engine.mcp")

mcp = MCPServer("context-engine")

# The agent and its checkpointer are created on first use and reused for the
# lifetime of the server process.
_stack = AsyncExitStack()
_agent = None
_agent_lock = asyncio.Lock()


async def _get_agent():
    global _agent
    async with _agent_lock:
        if _agent is None:
            saver = await _stack.enter_async_context(open_checkpointer())
            _agent = compile_agent(saver)
    return _agent


@mcp.tool()
async def ask_context_engine(query: str, session_id: str = "mcp-default-session") -> str:
    """Ask ContextEngine a question about the indexed documents and code.

    The agent routes the question to the vector index, the web, or a direct answer,
    grades what it retrieves, retries with a rewritten query when retrieval misses, and
    keeps multi-turn memory per session_id.

    Args:
        query: The question to answer.
        session_id: Conversation id; reuse it to keep context across calls.
    """
    logger.info("ask_context_engine on session %s", session_id)
    try:
        agent = await _get_agent()
        state = await agent.ainvoke(
            {"messages": [HumanMessage(content=query)]},
            config={"configurable": {"thread_id": session_id}},
        )
    except Exception as err:
        logger.exception("Agent run failed")
        return f"ContextEngine failed to answer: {err}"

    answer = state.get("final_output", "")
    sources = state.get("sources", [])
    if sources:
        answer += "\n\nSources:\n" + "\n".join(f"- {s}" for s in sources)
    return answer


@mcp.tool()
async def index_local_knowledge_base(directory_path: str) -> str:
    """Index the .md, .txt and .py files in a local folder (rebuilds the collection).

    Args:
        directory_path: Absolute path, or a path relative to the project root.
    """
    try:
        result = await asyncio.to_thread(ingest_directory, directory_path)
    except IngestionError as err:
        return f"Error: {err}"
    except Exception as err:
        logger.exception("Local ingestion failed")
        return f"Indexing failed: {err}"
    return f"Success! {result.message}"


@mcp.tool()
async def index_remote_github_repository(repository_url: str, branch_name: str = "main") -> str:
    """Clone a public Git repository over https and add its source files to the index.

    Args:
        repository_url: https URL of the repository, e.g. https://github.com/owner/repo
        branch_name: Branch to index. Defaults to 'main'.
    """
    try:
        result = await asyncio.to_thread(ingest_github_repository, repository_url, branch_name)
    except IngestionError as err:
        return f"Error: {err}"
    except Exception as err:
        logger.exception("Repository ingestion failed")
        return f"Indexing failed: {err}"
    return f"Success! {result.message}"


if __name__ == "__main__":
    logger.info("ContextEngine MCP server starting on stdio")
    mcp.run(transport="stdio")
