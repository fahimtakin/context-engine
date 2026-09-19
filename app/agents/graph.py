# app/agents/graph.py
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.redis.aio import AsyncRedisSaver 
from app.agents.state import AgentState
from app.agents.nodes import router_node, vector_retriever_node, web_search_node, generation_node
from app.core.config import settings

def route_conditional(state: AgentState) -> str:
    """Evaluates router logic parameters to pick the next step."""
    decision = state.get("router_decision")
    if decision == "vector_db":
        return "vector_db"
    elif decision == "web_search":
        return "web_search"
    return "direct"

def create_agent_workflow() -> StateGraph:
    """Builds the coordinated nodes state machine without global import side effects."""
    workflow = StateGraph(AgentState)

    workflow.add_node("router", router_node)
    workflow.add_node("vector_db", vector_retriever_node)
    workflow.add_node("web_search", web_search_node)
    workflow.add_node("generator", generation_node)

    workflow.set_entry_point("router")

    workflow.add_conditional_edges(
        "router",
        route_conditional,
        {
            "vector_db": "vector_db",
            "web_search": "web_search",
            "direct": "generator"
        }
    )

    workflow.add_edge("vector_db", "generator")
    workflow.add_edge("web_search", "generator")
    workflow.add_edge("generator", END)
    return workflow

def get_redis_saver() -> AsyncRedisSaver:
    """Instantiates a context-managed asynchronous checkpointer channel link to Redis Stack."""
    redis_url = f"redis://{settings.REDIS_HOST}:{settings.REDIS_PORT}"
    return AsyncRedisSaver.from_conn_string(redis_url)

async def get_runtime_agent_brain(saver: AsyncRedisSaver):
    """Compiles workflow maps bound to an active memory checkpointer instance."""
    workflow = create_agent_workflow()
    return workflow.compile(checkpointer=saver)
