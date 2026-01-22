from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Callable, Iterable, List, TypeVar

T = TypeVar("T")
R = TypeVar("R")


def default_n_jobs(n_jobs: int | None = None) -> int:
    """Return sensible default number of worker processes."""
    if n_jobs is None or n_jobs <= 0:
        return max(1, (os.cpu_count() or 2) - 1)
    return int(max(1, n_jobs))


def parallel_map(
    items: Iterable[T],
    fn: Callable[[T], R],
    n_jobs: int | None = None,
) -> List[R]:
    """Evaluate fn over items using a process pool."""
    items = list(items)
    n_jobs = default_n_jobs(n_jobs)

    if n_jobs == 1 or len(items) <= 1:
        return [fn(x) for x in items]

    results: List[R] = []
    with ProcessPoolExecutor(max_workers=n_jobs) as ex:
        fut_to_item = {ex.submit(fn, item): item for item in items}
        for fut in as_completed(fut_to_item):
            results.append(fut.result())
    return results
