import json

import pytest

from evaluation.metrics import first_relevant_rank, hit, mean, precision, reciprocal_rank
from evaluation.run_eval import DATASET, load_dataset


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
