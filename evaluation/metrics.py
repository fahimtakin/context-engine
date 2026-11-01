"""Retrieval metrics. Pure functions, so they are unit-tested without any services."""

import math
from collections.abc import Iterable, Sequence


def first_relevant_rank(retrieved: Sequence[str], expected: Iterable[str]) -> int | None:
    """1-based rank of the first retrieved source that is an expected source."""
    wanted = set(expected)
    for rank, source in enumerate(retrieved, start=1):
        if source in wanted:
            return rank
    return None


def hit(retrieved: Sequence[str], expected: Iterable[str]) -> bool:
    return first_relevant_rank(retrieved, expected) is not None


def reciprocal_rank(retrieved: Sequence[str], expected: Iterable[str]) -> float:
    rank = first_relevant_rank(retrieved, expected)
    return 1.0 / rank if rank else 0.0


def precision(retrieved: Sequence[str], expected: Iterable[str]) -> float | None:
    """Share of retrieved chunks that come from an expected source (None if nothing retrieved)."""
    if not retrieved:
        return None
    wanted = set(expected)
    return sum(source in wanted for source in retrieved) / len(retrieved)


def mean(values: Iterable[float | None]) -> float | None:
    kept = [v for v in values if v is not None]
    return sum(kept) / len(kept) if kept else None


def percentile(values: Iterable[float], q: float) -> float | None:
    """Nearest-rank percentile (q in 0-100) of the values, or None if there are none."""
    ordered = sorted(values)
    if not ordered:
        return None
    rank = max(1, math.ceil(q / 100 * len(ordered)))
    return ordered[rank - 1]


def pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value * 100:.1f}%"
