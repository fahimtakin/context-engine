# app/mcp/server.py
import sys
import os

# 1. Force Python to prioritize looking at your project root folder for imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 2. Match the working directory to the root so .env file maps correctly
os.chdir(project_root)

from mcp.server.mcpserver import MCPServer
from langchain_core.messages import HumanMessage
from app.agents.graph import get_runtime_agent_brain, get_redis_saver
from app.services.ingestion import ingest_directory
from app.services.git_ingestion import ingest_github_repository


# Initialize the modern MCPServer instance
mcp = MCPServer("context-engine")

@mcp.tool()
async def ask_context_engine(query: str, session_id: str = "mcp-default-session") -> str:
    """
    Queries ContextEngine's multi-agent system. 
    Automatically routes queries between local documentation indices and web search,
    preserving multi-turn conversation memory through a Redis session state store.
    
    Args:
        query: The natural language question or command to evaluate.
        session_id: The tracking key for continuous conversation context.
    """
    config = {
        "configurable": {"thread_id": session_id}
    }
    
    inputs = {
        "messages": [HumanMessage(content=query)]
    }
    
    # 🌟 PROTOCOL GUARD: Redirect standard prints out of stdout to prevent JSON-RPC corruption
    print(f"MCP Tool Received Query: '{query}' on session '{session_id}'", file=sys.stderr, flush=True)
    
    try:
        async with get_redis_saver() as saver:
            # Setup the checkpointer schemas cleanly
            await saver.asetup()
            runtime_brain = await get_runtime_agent_brain(saver)
            
            print("Executing LangGraph state routing nodes...", file=sys.stderr, flush=True)
            final_state = await runtime_brain.ainvoke(inputs, config=config)
            
            return final_state.get("final_output", "Error: No response generated from graph engine.")
            
    except Exception as e:
        error_msg = f"Execution failed inside ContextEngine multi-agent core pipeline: {str(e)}"
        print(f"{error_msg}", file=sys.stderr, flush=True)
        return error_msg

@mcp.tool()
async def index_local_knowledge_base(directory_path: str) -> str:
    """
    Triggers document parsing and vector pipeline storage over a specified local path target.
    
    Args:
        directory_path: Absolute or relative string path pointing to documentation folders.
    """
    print(f"📥 MCP Ingestion Request for Path: '{directory_path}'", file=sys.stderr, flush=True)
    
    if not os.path.exists(directory_path):
        return f"Error: Target path folder path '{directory_path}' could not be located on disk."
        
    try:
        ingest_directory(directory_path)
        return f"Success! ContextEngine indexed documents from target path source pipeline."
    except Exception as e:
        error_msg = f"Failed to ingest knowledge directory into Qdrant collection pipeline: {str(e)}"
        print(f"{error_msg}", file=sys.stderr, flush=True)
        return error_msg



@mcp.tool()
async def index_remote_github_repository(repository_url: str, branch_name: str = "main") -> str:
    """
    Clones any target public GitHub repository, splits its code structures into semantic chunks,
    generates vector embeddings, and writes the nodes directly into ContextEngine's Qdrant index storage layout.
    
    Args:
        repository_url: The absolute target URL string of the repository (e.g. 'https://github.com').
        branch_name: The development track target to read. Defaults to 'main'.
    """
    try:
        result = ingest_github_repository(repository_url, branch=branch_name)
        return result
    except Exception as e:
        return f"Failed to ingest remote repository target metrics layout: {str(e)}"


if __name__ == "__main__":
    # Launch stdio channel hooks directly
    print("ContextEngine MCP Server Process Initialized Successfully Over Stdio Channel", file=sys.stderr, flush=True)
    mcp.run(transport="stdio")
