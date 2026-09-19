# app/agents/state.py
from typing import List, Annotated, TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

class AgentState(TypedDict):
    """Tracks the continuous state through our multi-agent graph system."""
    
    messages: Annotated[List[BaseMessage], add_messages]      
    router_decision: str              
    retrieved_docs: List[str]         
    final_output: str                 
