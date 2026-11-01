from typing import Annotated, Literal, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

Route = Literal["vector_db", "web_search", "direct"]


class RetrievedDoc(TypedDict):
    source: str
    content: str


class AgentState(TypedDict, total=False):
    """State carried through the graph for one conversation turn.

    `messages` accumulates across turns (it is checkpointed per thread). Every other
    field is per-turn and is reset by the router at the start of each turn, so context
    retrieved for an earlier question can never leak into a later answer.
    """

    messages: Annotated[list[BaseMessage], add_messages]
    question: str  # the user's latest message
    search_query: str  # what we actually search with; may be rewritten by the agent
    router_decision: Route
    retrieved_docs: list[RetrievedDoc]
    retry_count: int  # number of query rewrites performed this turn
    used_web_fallback: bool  # True when vector retrieval missed and web search was used
    sources: list[str]
    final_output: str
