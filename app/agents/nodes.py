"""Graph nodes for the ContextEngine agent.

Turn flow (corrective RAG):

    router ─┬─ vector_db ─> retrieve ─> grade ─┬─ relevant docs ──────────────> generate
            │                                  ├─ none, retries left ─> rewrite ─> retrieve
            │                                  └─ none, no retries ───> web_search ─> generate
            ├─ web_search ──────────────────────────────────────────> web_search ─> generate
            └─ direct ──────────────────────────────────────────────────────────> generate
"""

import logging
import re
from typing import Literal

from duckduckgo_search import DDGS
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from app.agents.state import AgentState, RetrievedDoc
from app.core.config import settings
from app.core.llm import get_llm
from app.db.qdrant_client import get_vector_store

logger = logging.getLogger(__name__)

LLM_UNAVAILABLE_MESSAGE = (
    "The language model is unavailable right now, so I can't write an answer. "
    "Check that Ollama is running and reachable."
)


# --------------------------------------------------------------------------- schemas


class RouteDecision(BaseModel):
    choice: Literal["vector_db", "web_search", "direct"] = Field(
        description="vector_db for questions about the indexed documents or code, "
        "web_search for current events or general facts not in the index, "
        "direct for greetings and small talk."
    )


class GradeDecision(BaseModel):
    relevant: bool = Field(description="True if the document helps answer the question.")


class RewrittenQuery(BaseModel):
    query: str = Field(description="A clearer search query for the same question.")


# --------------------------------------------------------------------------- prompts

ROUTER_PROMPT = (
    "You route questions for an assistant that answers from an indexed knowledge base of "
    "documents and source code. Choose exactly one route:\n"
    "- vector_db: the question is about the indexed documents, a codebase, files, functions, "
    "architecture, or anything that could be answered from them.\n"
    "- web_search: the question needs current or external information (news, releases, "
    "prices, general world facts) that a private knowledge base would not contain.\n"
    "- direct: greetings, thanks, or small talk that needs no lookup.\n"
    "When unsure, choose vector_db."
)

GRADER_PROMPT = (
    "You check whether a retrieved document is relevant to a user's question. "
    "Answer relevant=true if the document contains information that helps answer the "
    "question, even partially. Answer relevant=false if it is about something else."
)

REWRITE_PROMPT = (
    "A search over a knowledge base of documents and source code returned nothing relevant "
    "for the question below. Rewrite it as a better search query: use concrete keywords, "
    "likely file, function or component names, and drop filler words. Keep the meaning."
)

GENERATOR_PROMPT = (
    "You are ContextEngine, an assistant that answers questions about the user's indexed "
    "documents and codebases.\n"
    "- When context is provided, base your answer on it and mention the source paths you used.\n"
    "- If the context does not contain the answer, say so plainly instead of guessing.\n"
    "- When no context is provided (greetings or small talk), reply briefly and naturally.\n\n"
    "Context:\n{context}"
)

_SMALL_TALK = {
    "hello",
    "hi",
    "hey",
    "greetings",
    "yo",
    "sup",
    "good morning",
    "good afternoon",
    "good evening",
    "thanks",
    "thank you",
    "who are you",
}


# --------------------------------------------------------------------------- helpers


def latest_user_message(state: AgentState) -> str:
    for message in reversed(state.get("messages", [])):
        if isinstance(message, HumanMessage):
            return str(message.content)
    return ""


def is_small_talk(text: str) -> bool:
    """Match greetings as whole words, so 'high availability' is not mistaken for 'hi'."""
    cleaned = re.sub(r"[^\w\s]", "", text.lower()).strip()
    return any(cleaned == phrase or cleaned.startswith(phrase + " ") for phrase in _SMALL_TALK)


def to_retrieved(doc: Document) -> RetrievedDoc:
    meta = doc.metadata or {}
    source = meta.get("source") or meta.get("file_path") or "unknown"
    repo = meta.get("repo_url")
    if repo:
        source = f"{repo}:{source}"
    return {"source": str(source), "content": doc.page_content}


def format_context(docs: list[RetrievedDoc]) -> str:
    if not docs:
        return "(no context)"
    return "\n\n".join(f"[Source: {d['source']}]\n{d['content']}" for d in docs)


def unique_sources(docs: list[RetrievedDoc]) -> list[str]:
    return list(dict.fromkeys(d["source"] for d in docs))


def search_web(query: str, max_results: int) -> list[RetrievedDoc]:
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
    except Exception:
        logger.warning("Web search failed for %r", query, exc_info=True)
        return []
    return [
        {"source": r.get("href") or "web", "content": f"{r.get('title', '')}\n{r.get('body', '')}"}
        for r in results
    ]


# --------------------------------------------------------------------------- nodes


def router_node(state: AgentState) -> dict:
    question = latest_user_message(state)
    # Reset every per-turn field so nothing from the previous turn leaks into this one.
    turn = {
        "question": question,
        "search_query": question,
        "retrieved_docs": [],
        "retry_count": 0,
        "used_web_fallback": False,
        "sources": [],
    }

    if is_small_talk(question):
        return {**turn, "router_decision": "direct"}

    try:
        decision = (
            get_llm()
            .with_structured_output(RouteDecision)
            .invoke([SystemMessage(content=ROUTER_PROMPT), HumanMessage(content=question)])
            .choice
        )
    except Exception:
        logger.warning("Router LLM call failed; defaulting to vector_db", exc_info=True)
        decision = "vector_db"

    logger.info("Routed question to %s", decision)
    return {**turn, "router_decision": decision}


def retrieve_node(state: AgentState) -> dict:
    query = state.get("search_query") or state.get("question", "")
    try:
        docs = get_vector_store().similarity_search(query, k=settings.RETRIEVAL_K)
    except Exception:
        logger.warning("Vector search failed for %r", query, exc_info=True)
        docs = []
    return {"retrieved_docs": [to_retrieved(d) for d in docs]}


def grade_documents_node(state: AgentState) -> dict:
    """Keep only the retrieved documents the LLM judges relevant to the question."""
    docs = state.get("retrieved_docs", [])
    if not docs:
        return {"retrieved_docs": []}

    question = state.get("question", "")
    try:
        grader = get_llm().with_structured_output(GradeDecision)
    except Exception:
        logger.warning("Grader unavailable; keeping all retrieved documents", exc_info=True)
        return {"retrieved_docs": docs}

    prompts = [
        [
            SystemMessage(content=GRADER_PROMPT),
            HumanMessage(content=f"Question: {question}\n\nDocument:\n{doc['content']}"),
        ]
        for doc in docs
    ]
    # Grade concurrently; results come back in input order, errors as values.
    verdicts = grader.batch(
        prompts, config={"max_concurrency": settings.GRADER_CONCURRENCY}, return_exceptions=True
    )

    relevant: list[RetrievedDoc] = []
    for doc, verdict in zip(docs, verdicts, strict=True):
        if isinstance(verdict, Exception):
            # Fail open: an unavailable grader should not throw away possibly good context.
            logger.warning("Grading failed for %s; keeping it", doc["source"], exc_info=verdict)
            keep = True
        else:
            keep = verdict.relevant
        if keep:
            relevant.append(doc)

    logger.info("Grading kept %d of %d documents", len(relevant), len(docs))
    return {"retrieved_docs": relevant}


def rewrite_query_node(state: AgentState) -> dict:
    question = state.get("question", "")
    try:
        rewritten = (
            get_llm()
            .with_structured_output(RewrittenQuery)
            .invoke([SystemMessage(content=REWRITE_PROMPT), HumanMessage(content=question)])
            .query.strip()
        )
    except Exception:
        logger.warning("Query rewrite failed; retrying with the original question", exc_info=True)
        rewritten = ""

    retry_count = state.get("retry_count", 0) + 1
    logger.info("Rewrite %d: %r -> %r", retry_count, question, rewritten or question)
    return {"search_query": rewritten or question, "retry_count": retry_count}


def web_search_node(state: AgentState) -> dict:
    query = state.get("search_query") or state.get("question", "")
    results = search_web(query, settings.WEB_SEARCH_RESULTS)
    return {
        "retrieved_docs": results,
        "used_web_fallback": state.get("router_decision") == "vector_db",
    }


def generation_node(state: AgentState) -> dict:
    docs = state.get("retrieved_docs", [])
    system = SystemMessage(content=GENERATOR_PROMPT.format(context=format_context(docs)))
    try:
        text = str(get_llm().invoke([system, *state.get("messages", [])]).content)
    except Exception:
        logger.error("Generation failed", exc_info=True)
        text = LLM_UNAVAILABLE_MESSAGE

    return {
        "messages": [AIMessage(content=text)],
        "final_output": text,
        "sources": unique_sources(docs),
    }


# --------------------------------------------------------------------------- edges


def route_after_router(state: AgentState) -> str:
    return state.get("router_decision", "vector_db")


def decide_after_grading(state: AgentState) -> str:
    """The self-correction step: answer, retry with a rewritten query, or fall back to the web."""
    if state.get("retrieved_docs"):
        return "generate"
    if state.get("retry_count", 0) < settings.MAX_RETRIEVAL_RETRIES:
        return "rewrite"
    return "web_search"
