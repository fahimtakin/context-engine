"""Shared fakes so the test suite runs offline: no Ollama, Qdrant, Redis or network."""

import pytest
from langchain_core.documents import Document
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import MemorySaver

from app.agents import nodes
from app.agents.graph import compile_agent
from app.agents.nodes import GradeDecision, RewrittenQuery, RouteDecision


class FakeLLM:
    """Scripted stand-in for ChatOllama.

    `grades` is consumed one value per grading call (True = relevant). When it runs
    out, documents are graded relevant.
    """

    def __init__(
        self,
        route="vector_db",
        grades=(),
        rewrite="rewritten query",
        answer="final answer",
        fail=False,
    ):
        self.route = route
        self.grades = list(grades)
        self.rewrite = rewrite
        self.answer = answer
        self.fail = fail
        self.structured_calls: list[str] = []
        self.generation_inputs: list[list] = []

    def with_structured_output(self, schema):
        llm = self

        class _Structured:
            def invoke(self, messages, **kwargs):
                llm.structured_calls.append(schema.__name__)
                if llm.fail:
                    raise ConnectionError("ollama is down")
                if schema is RouteDecision:
                    return RouteDecision(choice=llm.route)
                if schema is GradeDecision:
                    return GradeDecision(relevant=llm.grades.pop(0) if llm.grades else True)
                if schema is RewrittenQuery:
                    return RewrittenQuery(query=llm.rewrite)
                raise AssertionError(f"unexpected schema {schema}")

        return _Structured()

    def invoke(self, messages, **kwargs):
        if self.fail:
            raise ConnectionError("ollama is down")
        self.generation_inputs.append(messages)
        return AIMessage(content=self.answer)


class FakeVectorStore:
    """Returns the next batch of documents on each search and records the queries."""

    def __init__(self, batches=(), fail=False):
        self.batches = [list(b) for b in batches]
        self.fail = fail
        self.queries: list[str] = []

    def similarity_search(self, query, k=4):
        self.queries.append(query)
        if self.fail:
            raise ConnectionError("qdrant is down")
        return self.batches.pop(0) if self.batches else []


def doc(source: str, text: str = "some content") -> Document:
    return Document(page_content=text, metadata={"source": source})


@pytest.fixture
def wire(monkeypatch):
    """Install fakes into the node module. Returns a function so each test scripts its own."""

    def _wire(llm=None, store=None, web_results=None):
        llm = llm or FakeLLM()
        store = store or FakeVectorStore()
        web_calls: list[str] = []

        def fake_search_web(query, max_results):
            web_calls.append(query)
            return list(web_results or [{"source": "https://example.com", "content": "web hit"}])

        monkeypatch.setattr(nodes, "get_llm", lambda: llm)
        monkeypatch.setattr(nodes, "get_vector_store", lambda: store)
        monkeypatch.setattr(nodes, "search_web", fake_search_web)
        return llm, store, web_calls

    return _wire


@pytest.fixture
def agent():
    return compile_agent(MemorySaver())


def ask(agent, text, thread="t1"):
    from langchain_core.messages import HumanMessage

    return agent.invoke(
        {"messages": [HumanMessage(content=text)]},
        config={"configurable": {"thread_id": thread}},
    )
