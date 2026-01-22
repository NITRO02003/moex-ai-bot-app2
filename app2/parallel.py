from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor
from typing import Callable, Iterable, List, TypeVar

T = TypeVar("T")
R = TypeVar("R")


def _env_n_jobs() -> int | None:
    raw = os.getenv("APP2_N_JOBS") or os.getenv("MOEX_N_JOBS")
    if raw is None:
        return None
    try:
        val = int(raw)
    except ValueError:
        return None
    return val


def default_n_jobs(n_jobs: int | None = None) -> int:
    """Return default number of worker processes (env overrides)."""
    env_val = _env_n_jobs()
    if n_jobs is None or n_jobs <= 0:
        if env_val is not None:
            n_jobs = env_val
        else:
            n_jobs = (os.cpu_count() or 2) - 1
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
        results = list(ex.map(fn, items))
    return results
