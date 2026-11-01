"""Unit tests for individual nodes and routing helpers."""

import pytest
from langchain_core.documents import Document

from app.agents.nodes import (
    decide_after_grading,
    grade_documents_node,
    is_small_talk,
    router_node,
    to_retrieved,
    unique_sources,
)
from app.core.config import settings
from tests.conftest import FakeLLM


@pytest.mark.parametrize("text", ["hi", "Hello!", "hey there", "Good morning", "thanks!"])
def test_small_talk_is_detected(text):
    assert is_small_talk(text)


@pytest.mark.parametrize(
    "text",
    ["high availability setup?", "history of the repo", "hello_world.py contents", "heyday"],
)
def test_words_that_merely_start_with_a_greeting_are_not_small_talk(text):
    assert not is_small_talk(text)


def test_router_resets_per_turn_fields(wire):
    wire(llm=FakeLLM(route="web_search"))
    from langchain_core.messages import HumanMessage

    update = router_node(
        {
            "messages": [HumanMessage(content="latest news?")],
            "retrieved_docs": [{"source": "old", "content": "stale"}],
            "retry_count": 5,
        }
    )

    assert update["router_decision"] == "web_search"
    assert update["retrieved_docs"] == []
    assert update["retry_count"] == 0
    assert update["question"] == update["search_query"] == "latest news?"


def _docs(*sources):
    return [{"source": s, "content": f"content of {s}"} for s in sources]


def test_grading_keeps_relevant_docs_in_order(wire):
    wire(llm=FakeLLM(grades=[False, True, False, True]))
    update = grade_documents_node({"question": "q", "retrieved_docs": _docs("a", "b", "c", "d")})
    assert [d["source"] for d in update["retrieved_docs"]] == ["b", "d"]


def test_one_failed_grade_keeps_only_that_doc(wire):
    wire(llm=FakeLLM(grades=[False, ConnectionError("timeout"), False]))
    update = grade_documents_node({"question": "q", "retrieved_docs": _docs("a", "b", "c")})
    assert [d["source"] for d in update["retrieved_docs"]] == ["b"]


def test_decide_after_grading():
    docs = [{"source": "a", "content": "x"}]
    assert decide_after_grading({"retrieved_docs": docs}) == "generate"
    assert decide_after_grading({"retrieved_docs": [], "retry_count": 0}) == "rewrite"
    assert (
        decide_after_grading({"retrieved_docs": [], "retry_count": settings.MAX_RETRIEVAL_RETRIES})
        == "web_search"
    )


def test_to_retrieved_prefixes_repo_url():
    d = Document(page_content="x", metadata={"source": "src/app.py", "repo_url": "https://g/r"})
    assert to_retrieved(d)["source"] == "https://g/r:src/app.py"


def test_unique_sources_keeps_order():
    docs = [{"source": s, "content": ""} for s in ["b", "a", "b", "c"]]
    assert unique_sources(docs) == ["b", "a", "c"]
