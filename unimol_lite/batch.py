"""Parallel batch featurization built on :func:`smiles2unimolitem`.

Design notes (see ``_dev/分子采样性能优化/01-目标与设计.md`` §5.4):

- **equal length, input order**: results are collected by input index, so the
  output has exactly one entry per input SMILES (duplicates are *not* folded);
- **bounded submission**: a sliding window keeps at most ``window`` futures
  alive, so futures are never all materialized at once. The input list and the
  result list are still held in memory (both are ``N`` entries by contract);
  only the *in-flight* work is bounded;
- **``workers=1`` is a plain serial loop** (no thread-pool overhead at all);
- **per-row isolation**: an unexpected exception becomes
  ``(None, {"status": "error", "reason": <ExceptionType>})`` instead of aborting
  the batch. Internally every call asks for ``return_meta=True``, so
  ``on_timeout="error"`` is reported as ``status="timeout_error"`` on that row
  (the batch layer never raises for a single bad molecule). Invalid **batch**
  parameters (``coord_mode`` / ``on_timeout`` / ``timeout`` / ``workers`` /
  ``window``) raise ``ValueError`` before any work is submitted — a caller bug
  is not a per-row failure;
- **no new global state**: nothing module-level is mutated; a shared
  ``tokenizer`` is only read. RDKit embedding/MMFF release the GIL, which is
  why threads (not processes) are the right primitive here.
"""

from __future__ import annotations

import numbers
import os
import time
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, Dict, Iterable, List, Optional, Tuple

from .data.prepare_unimol_item import smiles2unimolitem
from .data.smiles2coords import _resolve_coord_mode, _validate_on_timeout, _validate_timeout
from .data.utils import _UNSET

#: ``min(8, os.cpu_count())`` — 8 threads is the measured sweet spot (16 threads
#: drop to ~52% efficiency, see the work order baseline table).
DEFAULT_MAX_WORKERS = 8

#: Batch-level default. ``smiles2unimolitem`` itself defaults to ``None`` so the
#: golden/legacy path stays untouched; the batch entry point is the place where a
#: finite budget protects the whole run.
DEFAULT_TIMEOUT = 30.0

#: One row: ``(item_or_None, meta)``.
BatchResult = Tuple[Optional[Dict[str, Any]], Dict[str, Any]]


def _default_workers() -> int:
    return min(DEFAULT_MAX_WORKERS, os.cpu_count() or 1)


def _resolve_workers(workers: Optional[int]) -> int:
    if workers is None:
        return _default_workers()
    if isinstance(workers, bool) or not isinstance(workers, numbers.Integral):
        raise ValueError(f"workers must be a positive integer, got {workers!r}")
    value = int(workers)
    if value <= 0:
        raise ValueError(f"workers must be a positive integer, got {workers!r}")
    return value


def _resolve_window(window: Optional[int], workers: int) -> int:
    if window is None:
        return max(workers * 4, 16)
    if isinstance(window, bool) or not isinstance(window, numbers.Integral):
        raise ValueError(f"window must be a positive integer, got {window!r}")
    value = int(window)
    if value <= 0:
        raise ValueError(f"window must be a positive integer, got {window!r}")
    return value


def _error_meta(reason: str, start: float) -> Dict[str, Any]:
    return dict(
        status="error",
        coord_mode=None,
        reason=reason,
        elapsed_s=time.perf_counter() - start,
    )


def _featurize_one(smiles: str, kwargs: Dict[str, Any]) -> BatchResult:
    """Featurize one SMILES, converting *any* exception into an error row."""
    start = time.perf_counter()
    try:
        return smiles2unimolitem(smiles, return_meta=True, **kwargs)
    except Exception as exc:  # noqa: BLE001 - isolation is the whole point
        return None, _error_meta(type(exc).__name__, start)


def _collect(future: Future) -> BatchResult:
    try:
        return future.result()
    except Exception as exc:  # pragma: no cover - defensive: _featurize_one guards
        return None, _error_meta(type(exc).__name__, time.perf_counter())


def featurize_batch(
    smiles: Iterable[str],
    *,
    workers: Optional[int] = None,
    timeout: Optional[float] = 30.0,
    on_timeout: str = "fallback_2d",
    coord_mode: str = "auto",
    rng_seed: int = 42,
    max_atoms: int = 256,
    tokenizer=None,
    window: Optional[int] = None,
) -> List[BatchResult]:
    """Featurize ``smiles`` in parallel, preserving input order and length.

    Args:
        smiles: iterable of SMILES strings (consumed once, duplicates kept).
        workers: thread count; ``None`` -> ``min(8, os.cpu_count() or 1)``;
            ``1`` runs the serial path; ``<= 0`` raises ``ValueError``.
        timeout: per-conformer wall-clock budget (embed + MMFF) forwarded to
            :func:`smiles2unimolitem`. Batch default ``30.0`` s; ``None`` turns
            the budget off (legacy untimed path).
        on_timeout: ``"fallback_2d"`` / ``"error"`` / ``"skip"``. With
            ``"error"`` a timeout is reported on that row as
            ``status="timeout_error"`` — it never aborts the batch.
        coord_mode: ``"auto"`` / ``"3d"`` / ``"2d"``, forwarded per row.
        rng_seed: seed for sampling / atom truncation, forwarded per row.
        max_atoms: atom cap, forwarded per row.
        tokenizer: optional shared tokenizer (read-only); the
            :func:`smiles2unimolitem` default is used when ``None``.
        window: bounded sliding-window size; ``None`` -> ``max(workers * 4, 16)``.

    Returns:
        A list with one ``(item | None, meta)`` entry per input SMILES, in input
        order. ``item`` is ``None`` for invalid SMILES, timeouts with
        ``on_timeout`` in ``{"error", "skip"}`` and isolated errors; ``meta``
        then carries ``status`` / ``reason``.

    Raises:
        ValueError: invalid ``workers`` / ``window`` / ``coord_mode`` /
            ``on_timeout`` / ``timeout`` — validated up front, before any row
            is submitted (an empty input still validates them).
    """
    batch = list(smiles)

    # validate up front: a bad parameter is a caller bug, not a per-row failure,
    # so it must raise instead of silently turning the whole batch into errors
    workers = _resolve_workers(workers)
    window = _resolve_window(window, workers)
    timeout = _validate_timeout(timeout)
    on_timeout = _validate_on_timeout(on_timeout)
    coord_mode = _resolve_coord_mode(coord_mode, _UNSET)
    if not batch:
        return []

    kwargs: Dict[str, Any] = dict(
        coord_mode=coord_mode,
        timeout=timeout,
        on_timeout=on_timeout,
        rng_seed=rng_seed,
        max_atoms=max_atoms,
    )
    if tokenizer is not None:
        kwargs["tokenizer"] = tokenizer

    results: List[Optional[BatchResult]] = [None] * len(batch)

    if workers == 1:
        for index, smi in enumerate(batch):
            results[index] = _featurize_one(smi, kwargs)
        return results  # type: ignore[return-value]

    with ThreadPoolExecutor(max_workers=workers) as pool:
        pending = deque()
        for index, smi in enumerate(batch):
            if len(pending) >= window:
                done_index, future = pending.popleft()
                results[done_index] = _collect(future)
            pending.append((index, pool.submit(_featurize_one, smi, kwargs)))
        while pending:
            done_index, future = pending.popleft()
            results[done_index] = _collect(future)

    return results  # type: ignore[return-value]
