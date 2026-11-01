import json

import pytest

from evaluation.judge import (
    GroundednessVerdict,
    RelevanceVerdict,
    judge_groundedness,
    judge_relevance,
)
from evaluation.metrics import (
    first_relevant_rank,
    hit,
    mean,
    percentile,
    precision,
    reciprocal_rank,
)
from evaluation.run_eval import DATASET, load_dataset


class StubJudge:
    """Returns a fixed verdict for whichever schema is requested, or fails."""

    def __init__(self, verdict=True, fail=False):
        self.verdict = verdict
        self.fail = fail
        self.prompts = []

    def with_structured_output(self, schema):
        judge = self

        class _Structured:
            def invoke(self, messages):
                judge.prompts.append(messages[-1].content)
                if judge.fail:
                    raise ConnectionError("ollama is down")
                if schema is GroundednessVerdict:
                    return GroundednessVerdict(grounded=judge.verdict)
                return RelevanceVerdict(relevant=judge.verdict)

        return _Structured()


def test_rank_hit_and_reciprocal_rank():
    retrieved = ["a.py", "b.py", "c.py"]
    assert first_relevant_rank(retrieved, {"b.py"}) == 2
    assert hit(retrieved, {"c.py"})
    assert not hit(retrieved, {"z.py"})
    assert reciprocal_rank(retrieved, {"b.py", "c.py"}) == pytest.approx(0.5)
    assert reciprocal_rank(retrieved, {"z.py"}) == 0.0


def test_precision_counts_chunks_and_handles_empty():
    assert precision(["a.py", "a.py", "b.py", "c.py"], {"a.py"}) == pytest.approx(0.5)
    assert precision([], {"a.py"}) is None


def test_mean_ignores_none():
    assert mean([1.0, None, 0.0]) == pytest.approx(0.5)
    assert mean([None]) is None


def test_percentile_nearest_rank():
    values = [5.0, 1.0, 3.0, 2.0, 4.0]
    assert percentile(values, 50) == 3.0
    assert percentile(values, 95) == 5.0
    assert percentile(values, 0) == 1.0
    assert percentile([], 50) is None


def test_judges_pass_context_and_return_verdicts():
    judge = StubJudge(verdict=False)
    docs = [{"source": "app/main.py", "content": "chat endpoint"}]

    assert judge_groundedness(judge, "It is in main.py", docs) is False
    assert judge_relevance(judge, "Where is chat?", "It is in main.py") is False
    assert "[Source: app/main.py]" in judge.prompts[0]
    assert "Where is chat?" in judge.prompts[1]


def test_failed_judge_returns_none_so_it_is_excluded():
    judge = StubJudge(fail=True)
    assert judge_groundedness(judge, "answer", []) is None
    assert judge_relevance(judge, "question", "answer") is None


def test_dataset_is_well_formed():
    items = load_dataset(DATASET)
    assert len(items) >= 20
    for item in items:
        assert item["route"] in {"vector_db", "web_search", "direct"}
        assert isinstance(item["expected_sources"], list)
        if item["route"] == "vector_db":
            assert item["expected_sources"], item["question"]


def test_expected_sources_exist_in_the_repo():
    from evaluation.run_eval import ROOT

    for item in load_dataset(DATASET):
        for source in item["expected_sources"]:
            assert (ROOT / source).is_file(), f"{source} missing for: {item['question']}"
    json.dumps(items := load_dataset(DATASET))  # serialisable
    assert items
