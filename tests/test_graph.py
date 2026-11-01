"""End-to-end runs of the agent graph with fake LLM, vector store and web search."""

from app.agents.nodes import LLM_UNAVAILABLE_MESSAGE, NOTHING_FOUND
from app.core.config import settings
from tests.conftest import FakeLLM, FakeVectorStore, ask, doc


def test_relevant_docs_on_first_try_go_straight_to_generation(wire, agent):
    llm, store, web_calls = wire(store=FakeVectorStore([[doc("app/main.py"), doc("README.md")]]))

    state = ask(agent, "How does the chat endpoint work?")

    assert state["router_decision"] == "vector_db"
    assert state["retry_count"] == 0
    assert state["sources"] == ["app/main.py", "README.md"]
    assert state["final_output"] == "final answer"
    assert state["used_web_fallback"] is False
    assert web_calls == []
    assert len(store.queries) == 1


def test_irrelevant_docs_trigger_a_rewrite_and_second_search(wire, agent):
    llm = FakeLLM(grades=[False, True], rewrite="chat endpoint FastAPI handler")
    batches = [[doc("docs/unrelated.md")], [doc("app/main.py")]]
    _, store, web_calls = wire(llm=llm, store=FakeVectorStore(batches))

    state = ask(agent, "where's the chat thing")

    assert store.queries == ["where's the chat thing", "chat endpoint FastAPI handler"]
    assert state["retry_count"] == 1
    assert state["sources"] == ["app/main.py"]
    assert web_calls == []


def test_falls_back_to_web_search_when_retries_are_exhausted(wire, agent):
    llm = FakeLLM(grades=[False, False])
    _, store, web_calls = wire(llm=llm, store=FakeVectorStore([[doc("a.md")], [doc("b.md")]]))

    state = ask(agent, "What is the latest LangGraph release?")

    assert len(store.queries) == 1 + settings.MAX_RETRIEVAL_RETRIES
    assert web_calls == ["rewritten query"]
    assert state["used_web_fallback"] is True
    assert state["sources"] == ["https://example.com"]


def test_retry_limit_is_respected(wire, agent, monkeypatch):
    monkeypatch.setattr(settings, "MAX_RETRIEVAL_RETRIES", 0)
    _, store, web_calls = wire(llm=FakeLLM(grades=[False]), store=FakeVectorStore([[doc("a.md")]]))

    state = ask(agent, "something not in the index")

    assert len(store.queries) == 1
    assert state["retry_count"] == 0
    assert len(web_calls) == 1


def test_loop_terminates_even_if_nothing_is_ever_relevant(wire, agent, monkeypatch):
    monkeypatch.setattr(settings, "MAX_RETRIEVAL_RETRIES", 3)
    llm = FakeLLM(grades=[False] * 10)
    batches = [[doc(f"{i}.md")] for i in range(10)]
    _, store, web_calls = wire(llm=llm, store=FakeVectorStore(batches))

    state = ask(agent, "unanswerable")

    assert len(store.queries) == 4
    assert state["retry_count"] == 3
    assert len(web_calls) == 1


def test_web_route_skips_the_vector_store(wire, agent):
    _, store, web_calls = wire(llm=FakeLLM(route="web_search"))

    state = ask(agent, "What's the weather in Halifax today?")

    assert state["router_decision"] == "web_search"
    assert store.queries == []
    assert len(web_calls) == 1
    assert state["used_web_fallback"] is False


def test_greeting_is_answered_directly_without_llm_routing(wire, agent):
    llm, store, web_calls = wire()

    state = ask(agent, "Hello!")

    assert state["router_decision"] == "direct"
    assert "RouteDecision" not in llm.structured_calls
    assert store.queries == [] and web_calls == []


def test_context_from_a_previous_turn_does_not_leak_into_the_next(wire, agent):
    llm, _, _ = wire(store=FakeVectorStore([[doc("app/main.py", "SECRET-CONTEXT")]]))

    first = ask(agent, "How does the chat endpoint work?", thread="memory")
    second = ask(agent, "thanks", thread="memory")

    assert first["sources"] == ["app/main.py"]
    assert second["sources"] == []
    assert second["retrieved_docs"] == []
    system_prompt = llm.generation_inputs[-1][0].content
    assert "SECRET-CONTEXT" not in system_prompt
    # ...while the conversation history itself is kept.
    assert len(second["messages"]) == 4


def test_generator_is_told_when_a_lookup_found_nothing(wire, agent):
    # Nothing in the index and the web search returns nothing either.
    llm, _, web_calls = wire(store=FakeVectorStore([[], []]), web_results=[])

    state = ask(agent, "How is billing implemented?")

    assert len(web_calls) == 1 and state["sources"] == []
    assert NOTHING_FOUND in llm.generation_inputs[-1][0].content


def test_small_talk_is_not_treated_as_a_failed_lookup(wire, agent):
    llm, _, _ = wire()

    ask(agent, "Hello!")

    assert NOTHING_FOUND not in llm.generation_inputs[-1][0].content


def test_survives_ollama_being_down(wire, agent):
    llm = FakeLLM(fail=True)
    _, store, _ = wire(llm=llm, store=FakeVectorStore([[doc("app/main.py")]]))

    state = ask(agent, "How does ingestion work?")

    assert state["router_decision"] == "vector_db"  # router fallback
    assert state["sources"] == ["app/main.py"]  # grader failed open
    assert state["final_output"] == LLM_UNAVAILABLE_MESSAGE


def test_survives_qdrant_being_down(wire, agent):
    _, store, web_calls = wire(store=FakeVectorStore(fail=True))

    state = ask(agent, "How does ingestion work?")

    assert state["used_web_fallback"] is True
    assert len(web_calls) == 1
    assert state["final_output"] == "final answer"
