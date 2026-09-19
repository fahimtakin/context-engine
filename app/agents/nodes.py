# app/agents/nodes.py
import json
import os
import sys
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage 
from langchain_qdrant import QdrantVectorStore
from langchain_ollama import OllamaEmbeddings
from duckduckgo_search import DDGS 
from app.core.llm import MockLLM, get_llm, resolve_ollama_endpoint
from app.core.config import settings
from app.db.qdrant_client import COLLECTION_NAME
from app.agents.state import AgentState

class RouteDecision(BaseModel):
    choice: str = Field(description="Must be exactly 'vector_db', 'web_search', or 'direct'")

def get_embeddings_engine() -> OllamaEmbeddings:
    resolved_url = resolve_ollama_endpoint()
    return OllamaEmbeddings(base_url=resolved_url, model="mxbai-embed-large")

def get_vector_store() -> QdrantVectorStore:
    """
    Safely retrieves the active Qdrant vector store collection.
    Dynamically maps host names depending on the running container state.
    """
    if os.path.exists("/.dockerenv") or os.environ.get("DOCKER_CONTAINER") == "true":
        host_target = "qdrant"
    else:
        host_target = settings.QDRANT_HOST
        
    port_target = settings.QDRANT_PORT 
    qdrant_url = f"http://{host_target}:{port_target}"
    embeddings = get_embeddings_engine()
    
    return QdrantVectorStore.from_existing_collection(
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        url=qdrant_url,
        timeout=30.0 
    )

def custom_web_search(query: str) -> str:
    try:
        with DDGS() as ddgs:
            results = [r for r in ddgs.text(query, max_results=3)]
            if not results:
                return "No search results found."
            return "\n\n".join([f"Title: {r['title']}\nSnippet: {r['body']}" for r in results])
    except Exception as e:
        return f"Web search tool execution failed: {str(e)}"

def router_node(state: AgentState) -> dict:
    last_message = state["messages"][-1].content
    cleaned_msg = str(last_message).lower().strip().replace("?", "").replace("!", "")
    
    # Absolute conversational greetings guard to prevent routing simple greetings to Web Search
    greetings = {
        "hello", "hi", "hey", "greetings", "yo", "sup", "good morning", 
        "good afternoon", "good evening", "test", "clear", "who are you"
    }
    
    if cleaned_msg in greetings or any(cleaned_msg.startswith(g) for g in greetings):
        return {"router_decision": "direct"}    
            
    decision = "vector_db"
    try:
        llm = get_llm()
        try:
            structured_llm = llm.with_structured_output(RouteDecision)
            result = structured_llm.invoke([
                SystemMessage(content="Classify query into: 'vector_db', 'web_search', or 'direct'"),
                HumanMessage(content=last_message)
            ])
            decision = result.choice
        except Exception as net_err:
            print(f"Hostname resolution dropped during routing. Activating Mock Structured fallback. Error: {str(net_err)}", flush=True)
            mock_engine = MockLLM().with_structured_output(RouteDecision)
            result = mock_engine.invoke([HumanMessage(content=last_message)])
            decision = result.choice
            
    except Exception as e:
        print(f"Top level router fallback activated. Defaulting to vector_db. Error: {str(e)}", flush=True)
        decision = "vector_db"
        
    return {"router_decision": decision}

def vector_retriever_node(state: AgentState) -> dict:
    try:
        last_message = state["messages"][-1].content
        vector_store = get_vector_store()
        docs = vector_store.similarity_search(last_message, k=4)
        formatted_docs = [f"Source File Path: {d.metadata.get('source')}\nOrigin Repo: {d.metadata.get('repo_url')}\nContent:\n{d.page_content}" for d in docs]
    except Exception as e:
        print(f"Vector db extraction drop caught: {str(e)}", flush=True)
        formatted_docs = ["Source: Database Fallback\nContent: (Vector storage container interface timed out)"]

    return {"retrieved_docs": formatted_docs}

def web_search_node(state: AgentState) -> dict:
    last_message = state["messages"][-1].content
    results = custom_web_search(last_message)
    return {"retrieved_docs": [f"Source: Web Search Results\nContent: {results}"]}

def generation_node(state: AgentState) -> dict:
    context = "\n\n".join(state.get("retrieved_docs", []))
    
    system_prompt = (
        "You are ContextEngine, a strict full-stack AI engineering assistant analyzing custom indexed files.\n"
        "Analyze the user's request using ONLY the literal context blocks provided below. "
        "CRITICAL RULE: Do not use any outside knowledge about real-world biology, books, or internet definitions. "
        "If the user asks for an analysis, base your interpretation exclusively on the unique narrative details "
        "found in the text (e.g., the cup of tea, the sack-like curtain, the cabin dangling from a branch, the silky orange and black wings, and the duties of the shadows).\n\n"
        f"Provided Context Blocks:\n{context}"
    )
    
    full_messages = [SystemMessage(content=system_prompt)] + state["messages"]
    llm = get_llm()
    
    try:
        response = llm.invoke(full_messages)
        final_text = response.content
    except Exception as e:
        print(f"Ollama generation fallback triggered: {str(e)}", flush=True)
        final_text = f"ContextEngine local execution fallback. Context data:\n{context}"
        
    return {
        "messages": [AIMessage(content=final_text)], 
        "final_output": final_text
    }