"""Measure ContextEngine's routing and retrieval against a fixed question set.

It indexes this repository's own code and docs into a separate Qdrant collection
(so your real index is untouched), then for every question it records:

  * router accuracy  - did the router pick the labelled route?
  * baseline         - plain top-k vector search: hit rate, MRR, precision@k
  * corrective       - the agent's grade -> rewrite -> re-retrieve loop: hit rate,
                       precision of the kept context, and how often it fell back to the web

Requires Ollama (llama3.1 + mxbai-embed-large) and Qdrant to be running.

    python -m evaluation.run_eval                 # index + evaluate
    python -m evaluation.run_eval --skip-index    # reuse the existing eval collection
"""

import argparse
import json
import logging
import time
from datetime import UTC, datetime
from pathlib import Path

from langchain_core.messages import HumanMessage

from app.agents.nodes import (
    decide_after_grading,
    generation_node,
    grade_documents_node,
    retrieve_node,
    rewrite_query_node,
    router_node,
    to_retrieved,
)
from app.agents.state import RetrievedDoc
from app.core.config import settings
from app.core.llm import get_llm
from app.core.logging_setup import configure_logging
from app.db.qdrant_client import get_vector_store
from app.services.ingestion import IGNORED_DIRS, ingest_directory
from evaluation.judge import judge_groundedness, judge_relevance
from evaluation.metrics import hit, mean, pct, percentile, precision, reciprocal_rank

ROOT = Path(__file__).resolve().parent.parent
DATASET = Path(__file__).with_name("dataset.jsonl")
RESULTS = Path(__file__).with_name("results.json")
EVAL_COLLECTION = "context_engine_eval"
# Keep tests and the eval set itself out of the index so answers can't leak in.
EVAL_IGNORED = IGNORED_DIRS | {"tests", "evaluation", "context-engine-ui", "documents"}

logger = logging.getLogger("evaluation")


def load_dataset(path: Path = DATASET) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def run_corrective(question: str) -> tuple[list[RetrievedDoc], bool, int]:
    """Run the agent's retrieval loop exactly as the graph does, minus generation."""
    state: dict = {"question": question, "search_query": question, "retry_count": 0}
    while True:
        state |= retrieve_node(state)
        state |= grade_documents_node(state)
        step = decide_after_grading(state)
        if step != "rewrite":
            break
        state |= rewrite_query_node(state)
    return state.get("retrieved_docs", []), step == "web_search", state.get("retry_count", 0)


def evaluate(items: list[dict], judge_answers: bool = True) -> dict:
    store = get_vector_store()
    llm = get_llm()
    rows = []
    for item in items:
        q, expected = item["question"], item["expected_sources"]
        started = time.perf_counter()
        route = router_node({"messages": [HumanMessage(content=q)]})["router_decision"]
        agent_seconds = time.perf_counter() - started
        row = {"question": q, "expected_route": item["route"], "route": route}

        if item["route"] == "vector_db":
            baseline = [
                to_retrieved(d)["source"]
                for d in store.similarity_search(q, k=settings.RETRIEVAL_K)
            ]
            t = time.perf_counter()
            docs, fell_back, retries = run_corrective(q)
            agent_seconds += time.perf_counter() - t
            corrective = [d["source"] for d in docs]
            if judge_answers:
                t = time.perf_counter()
                answer = generation_node(
                    {"messages": [HumanMessage(content=q)], "retrieved_docs": docs}
                )["final_output"]
                agent_seconds += time.perf_counter() - t
                row |= {
                    "answer": answer,
                    "grounded": judge_groundedness(llm, answer, docs),
                    "answer_relevant": judge_relevance(llm, q, answer),
                    "agent_seconds": round(agent_seconds, 2),
                }
            row |= {
                "baseline_sources": baseline,
                "baseline_hit": hit(baseline, expected),
                "baseline_rr": reciprocal_rank(baseline, expected),
                "baseline_precision": precision(baseline, expected),
                "corrective_sources": corrective,
                "corrective_hit": hit(corrective, expected),
                "corrective_precision": precision(corrective, expected),
                "web_fallback": fell_back,
                "retries": retries,
            }
        row["seconds"] = round(time.perf_counter() - started, 2)
        rows.append(row)
        logger.info("%-75s route=%s", q[:75], route)

    retrieval = [r for r in rows if r["expected_route"] == "vector_db"]
    answered = [r for r in retrieval if "answer" in r]
    latencies = [r["agent_seconds"] for r in answered]
    summary = {
        "questions": len(rows),
        "retrieval_questions": len(retrieval),
        "router_accuracy": mean(float(r["route"] == r["expected_route"]) for r in rows),
        "baseline_hit_rate": mean(float(r["baseline_hit"]) for r in retrieval),
        "baseline_mrr": mean(r["baseline_rr"] for r in retrieval),
        "baseline_precision": mean(r["baseline_precision"] for r in retrieval),
        "corrective_hit_rate": mean(float(r["corrective_hit"]) for r in retrieval),
        "corrective_precision": mean(r["corrective_precision"] for r in retrieval),
        "web_fallback_rate": mean(float(r["web_fallback"]) for r in retrieval),
        "avg_retries": mean(float(r["retries"]) for r in retrieval),
        "groundedness": mean(_as_float(r["grounded"]) for r in answered),
        "answer_relevance": mean(_as_float(r["answer_relevant"]) for r in answered),
        "latency_p50_seconds": percentile(latencies, 50),
        "latency_p95_seconds": percentile(latencies, 95),
    }
    return {"summary": summary, "rows": rows}


def _as_float(verdict: bool | None) -> float | None:
    return None if verdict is None else float(verdict)


def print_report(summary: dict) -> None:
    k = settings.RETRIEVAL_K
    print("\nContextEngine evaluation")
    print(f"  model={settings.LLM_MODEL}  embeddings={settings.EMBEDDING_MODEL}  k={k}")
    print(f"  questions={summary['questions']} (retrieval={summary['retrieval_questions']})\n")
    print(f"  Router accuracy                {pct(summary['router_accuracy'])}")
    print(f"  Baseline   hit@{k}               {pct(summary['baseline_hit_rate'])}")
    print(f"  Baseline   MRR                 {summary['baseline_mrr'] or 0:.3f}")
    print(f"  Baseline   context precision   {pct(summary['baseline_precision'])}")
    print(f"  Corrective hit rate            {pct(summary['corrective_hit_rate'])}")
    print(f"  Corrective context precision   {pct(summary['corrective_precision'])}")
    print(f"  Web fallback rate              {pct(summary['web_fallback_rate'])}")
    print(f"  Avg rewrites per question      {summary['avg_retries'] or 0:.2f}")
    if summary["latency_p50_seconds"] is not None:
        print(f"  Answer groundedness            {pct(summary['groundedness'])}")
        print(f"  Answer relevance               {pct(summary['answer_relevance'])}")
        p50, p95 = summary["latency_p50_seconds"], summary["latency_p95_seconds"]
        print(f"  Agent latency p50 / p95        {p50:.1f}s / {p95:.1f}s")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--skip-index", action="store_true", help="reuse the existing eval collection"
    )
    parser.add_argument(
        "--skip-answers",
        action="store_true",
        help="evaluate retrieval only; skip generating and judging answers",
    )
    parser.add_argument("--output", type=Path, default=RESULTS)
    args = parser.parse_args()

    configure_logging(settings.LOG_LEVEL)
    settings.QDRANT_COLLECTION = EVAL_COLLECTION

    if not args.skip_index:
        result = ingest_directory(ROOT, recreate=True, ignored_dirs=EVAL_IGNORED)
        logger.info(result.message)

    report = evaluate(load_dataset(), judge_answers=not args.skip_answers)
    report["meta"] = {
        "run_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "llm_model": settings.LLM_MODEL,
        "embedding_model": settings.EMBEDDING_MODEL,
        "k": settings.RETRIEVAL_K,
        "max_retrieval_retries": settings.MAX_RETRIEVAL_RETRIES,
    }
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print_report(report["summary"])
    print(f"Full results written to {args.output.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
