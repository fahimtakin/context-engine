import pytest
from fastapi.testclient import TestClient

import app.main as main
from app.services.ingestion import IngestionError, IngestionResult


class FakeAgent:
    def __init__(self, result=None, error=None):
        self.result = result or {}
        self.error = error
        self.calls = []

    async def ainvoke(self, inputs, config):
        self.calls.append((inputs, config))
        if self.error:
            raise self.error
        return self.result


@pytest.fixture
def make_client():
    def _make(agent=None):
        return TestClient(main.create_app(agent=agent or FakeAgent()))

    return _make


def test_health(make_client):
    with make_client() as client:
        assert client.get("/health").json() == {"status": "ok"}


def test_ready_reports_degraded_dependencies(make_client, monkeypatch):
    monkeypatch.setattr(main, "_qdrant_ok", lambda: False)

    async def ollama_ok():
        return True

    monkeypatch.setattr(main, "_ollama_ok", ollama_ok)
    with make_client() as client:
        response = client.get("/ready")
    assert response.status_code == 503
    assert response.json()["checks"]["qdrant"] is False


def test_chat_returns_answer_route_and_sources(make_client):
    agent = FakeAgent(
        {
            "final_output": "It is in app/main.py",
            "router_decision": "vector_db",
            "sources": ["app/main.py"],
            "retry_count": 1,
            "used_web_fallback": False,
        }
    )
    with make_client(agent) as client:
        body = client.post(
            "/api/chat", json={"message": "Where is chat?", "thread_id": "s1"}
        ).json()

    assert body == {
        "response": "It is in app/main.py",
        "thread_id": "s1",
        "router_decision": "vector_db",
        "sources": ["app/main.py"],
        "retries": 1,
        "used_web_fallback": False,
    }
    assert agent.calls[0][1] == {"configurable": {"thread_id": "s1"}}


def test_chat_generates_a_thread_id(make_client):
    with make_client(FakeAgent({"final_output": "hi"})) as client:
        body = client.post("/api/chat", json={"message": "hello"}).json()
    assert len(body["thread_id"]) == 36


def test_chat_rejects_empty_message(make_client):
    with make_client() as client:
        assert client.post("/api/chat", json={"message": ""}).status_code == 422


def test_chat_hides_internal_errors(make_client):
    with make_client(FakeAgent(error=RuntimeError("redis password=hunter2"))) as client:
        response = client.post("/api/chat", json={"message": "hello"})
    assert response.status_code == 500
    assert "hunter2" not in response.text


def test_index_git_success(make_client, monkeypatch):
    monkeypatch.setattr(
        main,
        "ingest_github_repository",
        lambda url, branch: IngestionResult(source=url, files=3, chunks=12),
    )
    with make_client() as client:
        response = client.post("/api/index-git", json={"repo_url": "https://github.com/o/r"})
    assert response.status_code == 200
    assert response.json()["chunks"] == 12


def test_index_git_reports_ingestion_errors_as_400(make_client, monkeypatch):
    def fail(url, branch):
        raise IngestionError("Could not clone https://github.com/o/missing")

    monkeypatch.setattr(main, "ingest_github_repository", fail)
    with make_client() as client:
        response = client.post("/api/index-git", json={"repo_url": "https://github.com/o/missing"})
    assert response.status_code == 400
    assert "Could not clone" in response.json()["detail"]
