"""Result caching and parallel experiment execution.

The experiments in this project are individually cheap but numerous: the paper
sweeps four activation functions, two optimisers, several architectures and
multiple random seeds.  A single 10 000-epoch run of the 12-10-7-5-4-3-2 network
takes a couple of minutes, almost all of it PyTorch dispatch overhead on very
small tensors -- which means threads do not help but *processes* scale nearly
linearly.

Two utilities:

* :func:`cached` memoises a result to disk, so re-running a notebook is instant
  while the first run still shows the real computation.
* :func:`parallel_map` farms independent configurations out to worker
  processes, each pinned to a single torch thread.

Cache entries are keyed by an explicit name.  Change the experiment, change the
name (or pass ``force=True``) -- the cache does not try to detect stale code.
"""

from __future__ import annotations

import gzip
import os
import pickle
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

__all__ = ["CACHE_DIR", "cached", "cache_path", "clear_cache", "parallel_map"]

_PKG_ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = _PKG_ROOT / "results"


def cache_path(name: str, directory: str | os.PathLike | None = None) -> Path:
    d = Path(directory) if directory is not None else CACHE_DIR
    return d / f"{name}.pkl.gz"


def cached(
    name: str,
    fn: Callable[[], Any],
    *,
    force: bool = False,
    directory: str | os.PathLike | None = None,
    verbose: bool = True,
) -> Any:
    """Return ``fn()``, caching the result on disk under ``name``.

    Parameters
    ----------
    force
        Recompute and overwrite an existing entry.
    """
    path = cache_path(name, directory)
    if path.exists() and not force:
        with gzip.open(path, "rb") as fh:
            obj = pickle.load(fh)
        if verbose:
            size = path.stat().st_size / 1e6
            print(f"[cache] loaded '{name}' ({size:.1f} MB)")
        return obj

    if verbose:
        print(f"[cache] computing '{name}' ...", flush=True)
    t0 = time.time()
    obj = fn()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with gzip.open(tmp, "wb", compresslevel=4) as fh:
        pickle.dump(obj, fh, protocol=pickle.HIGHEST_PROTOCOL)
    tmp.replace(path)
    if verbose:
        size = path.stat().st_size / 1e6
        print(f"[cache] stored '{name}' ({size:.1f} MB, {time.time() - t0:.0f}s)")
    return obj


def clear_cache(pattern: str = "*", directory: str | os.PathLike | None = None) -> int:
    """Delete cache entries matching a glob; returns the number removed."""
    d = Path(directory) if directory is not None else CACHE_DIR
    if not d.exists():
        return 0
    n = 0
    for p in d.glob(f"{pattern}.pkl.gz"):
        p.unlink()
        n += 1
    return n


# --------------------------------------------------------------------------- #
# parallel execution
# --------------------------------------------------------------------------- #
def _init_worker() -> None:
    """Pin each worker to one thread.

    The models here are tiny, so intra-op parallelism buys nothing (measured:
    14.0 s single-threaded versus 14.6 s on six threads) while oversubscribing
    cores would slow every worker down.
    """
    try:
        import torch

        torch.set_num_threads(1)
    except ImportError:  # pragma: no cover - torch is optional for some paths
        pass
    os.environ.setdefault("OMP_NUM_THREADS", "1")


def parallel_map(
    fn: Callable[..., Any],
    configs: Sequence[Any],
    max_workers: int | None = None,
    verbose: bool = True,
) -> list[Any]:
    """Run ``fn(config)`` for each config in a separate process.

    ``fn`` must be a module-level function (picklable), not a lambda or closure.
    """
    if max_workers is None:
        max_workers = min(len(configs), max(1, (os.cpu_count() or 2) - 1))
    t0 = time.time()
    if verbose:
        print(f"[parallel] {len(configs)} jobs on {max_workers} workers ...", flush=True)
    with ProcessPoolExecutor(max_workers=max_workers, initializer=_init_worker) as ex:
        results = list(ex.map(fn, configs))
    if verbose:
        print(f"[parallel] done in {time.time() - t0:.0f}s")
    return results
